#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <cstdio>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
#endif

#include "common.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "OrtNet.h"

std::string keys =
"{ help  h     | | Print help message. }"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU }"
"{ async       | 0 | Number of asynchronous forwards at the same time. "
"Choose 0 for synchronous mode }";

using namespace cv;
using namespace dnn;
using namespace std::chrono;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

void postprocess(Mat& frame, std::pair<float*, float*> outs);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

OrtNet &ortNet = OrtNet();


#ifdef CV_CXX11
template <typename T>
class QueueFPS : public std::queue<T>
{
public:
	QueueFPS() : counter(0) {}

	void push(const T& entry)
	{
		std::lock_guard<std::mutex> lock(mutex);

		std::queue<T>::push(entry);
		counter += 1;
		if (counter == 1)
		{
			// Start counting from a second frame (warmup).
			tm.reset();
			tm.start();
		}
	}

	T get()
	{
		std::lock_guard<std::mutex> lock(mutex);
		T entry = this->front();
		this->pop();
		return entry;
	}

	float getFPS()
	{
		tm.stop();
		double fps = counter / tm.getTimeSec();
		tm.start();
		return static_cast<float>(fps);
	}

	void clear()
	{
		std::lock_guard<std::mutex> lock(mutex);
		while (!this->empty())
			this->pop();
	}

	unsigned int counter;

private:
	TickMeter tm;
	std::mutex mutex;
};
#endif  // CV_CXX11

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	const std::string modelName = parser.get<String>("@alias");
	const std::string zooFile = parser.get<String>("zoo");

	keys += genPreprocArguments(modelName, zooFile);

	parser = CommandLineParser(argc, argv, keys);
	parser.about("Use this script to run object detection deep learning networks using OpenCV.");
	/*
	if (argc == 1 || parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	*/

	confThreshold = parser.get<float>("thr");
	nmsThreshold = parser.get<float>("nms");
	float scale = parser.get<float>("scale");
	Scalar mean = parser.get<Scalar>("mean");
	bool swapRB = parser.get<bool>("rgb");
	int inpWidth = parser.get<int>("width");
	int inpHeight = parser.get<int>("height");
	size_t async = parser.get<int>("async");
	// CV_Assert(parser.has("model"));
	std::string modelPath = findFile(parser.get<String>("model"));
	std::string configPath = findFile(parser.get<String>("config"));

	// Open file with classes names.
	if (parser.has("classes"))
	{
		std::string file = parser.get<String>("classes");
		std::ifstream ifs(file.c_str());
		if (!ifs.is_open())
			CV_Error(Error::StsError, "File " + file + " not found");
		std::string line;
		while (std::getline(ifs, line))
		{
			classes.push_back(line);
		}
	}

	ortNet.Init(L"D:/REPOs/ML/ssdIE/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx");

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	int initialConf = (int)(confThreshold * 100);
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	/*
	if (parser.has("input"))
		cap.open(parser.get<String>("input"));
	else
		cap.open(parser.get<int>("device"));
	*/
	cap.open("D:/REPOs/ML/data/VID_20190627_191450.mp4");

#ifdef CV_CXX11
	bool process = true;

	// Frames capturing thread
	QueueFPS<Mat> framesQueue;
	std::thread framesThread([&]() {
		Mat frame;
		while (process)
		{
			cap >> frame;
			// cv::flip(frame, frame, 1);
			if (!frame.empty())
				framesQueue.push(frame.clone());
			else
				break;

			// std::this_thread::sleep_for(std::chrono::milliseconds(30));
		}
		});

	// Frames processing thread
	QueueFPS<Mat> processedFramesQueue;
	QueueFPS<std::pair<float*, float*>> predictionsQueue;
	std::thread processingThread([&]() {
		while (process)
		{
			// Get a next frame
			Mat frame;
			if (!framesQueue.empty())
			{
				frame = framesQueue.get();
				framesQueue.clear();
			}

			// Process the frame
			if (!frame.empty())
			{
				ortNet.setInputTensor(frame);
				processedFramesQueue.push(frame);
				ortNet.forward();
				predictionsQueue.push(ortNet.getOuts());
				
			}
		}
		});

	// Postprocessing and rendering loop
	while (waitKey(1) < 0)
	{
		if (predictionsQueue.empty())
			continue;
		std::pair<float*, float*> outs = predictionsQueue.get();

		if (processedFramesQueue.empty())
			continue;

		Mat frame = processedFramesQueue.get();

		postprocess(frame, outs);

		if (predictionsQueue.counter > 1)
		{
			std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
			putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

			label = format("Network: %.2f FPS", predictionsQueue.getFPS());
			putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

			label = format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
			putText(frame, label, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		}

		imshow(kWinName, frame);
	}

	process = false;
	framesThread.join();
	processingThread.join();

#else  // CV_CXX11
	if (async)
		CV_Error(Error::StsNotImplemented, "Asynchronous forward is supported only with Inference Engine backend.");

	// Process frames.
	Mat frame, blob;
	while (waitKey(1) < 0)
	{
		cap >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}

		preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);

		std::vector<Mat> outs;
		net.forward(outs, outNames);

		postprocess(frame, outs, net);

		// Put efficiency information.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		imshow(kWinName, frame);
	}
#endif  // CV_CXX11
	return 0;
}


void postprocess(Mat& frame, std::pair<float*, float*> outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;

	auto ssd_scores = outs.first;
	auto ssd_boxes = outs.second;

	//CV_Assert(outs.first > 0);

	for (size_t i = 0; i < 3234; i++) {
		for (size_t j = 1; j < 78; j++) {
			float confidence = ssd_scores[78 * i + j];
			if (confidence > confThreshold)
			{
				int centerX = (int)(ssd_boxes[4*i] * frame.cols);
				int centerY = (int)(ssd_boxes[4*i + 1] * frame.rows);
				int width = (int)(ssd_boxes[4*i + 2] * frame.cols);
				int height = (int)(ssd_boxes[4*i + 3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back((int)j - 1);;
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

	std::string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
	confThreshold = pos * 0.01f;
}
