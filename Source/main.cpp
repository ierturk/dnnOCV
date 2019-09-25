#include <chrono>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include <mutex>
#include <thread>
#include <queue>

#include "OrtNet.h"

using namespace cv;
using namespace dnn;
using namespace std::chrono;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

void postprocess(Mat& frame, std::pair<float*, float*> outs);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

OrtNet *ortNet = new OrtNet();

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

int main(int argc, char** argv)
{
	confThreshold = 0.5;
	nmsThreshold = 0.4;

	ortNet->Init("/home/ierturk/Work/REPOs/ssd/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx");

	// Create a window
	static const std::string kWinName = "sgDetector";
	namedWindow(kWinName, WINDOW_NORMAL);
	int initialConf = (int)(confThreshold * 100);
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	cap.open("/home/ierturk/Downloads/happytime-rtsp-server/r.mp4");

	bool process = true;

	// Frames capturing thread
	QueueFPS<Mat> framesQueue;
	std::thread framesThread([&]() {
		Mat frame;
		while (process) {
			cap >> frame;
			if (!frame.empty())
				framesQueue.push(frame.clone());
			else
				break;
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
				ortNet->setInputTensor(frame);
				processedFramesQueue.push(frame);
				ortNet->forward();
				predictionsQueue.push(ortNet->getOuts());
				
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

		cv::resize(frame, frame, Size(512, 288));

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

                int left = (int)(ssd_boxes[4 * i] * frame.cols);
                int top = (int)(ssd_boxes[4 * i + 1] * frame.rows);
                int right = (int)(ssd_boxes[4 * i + 2] * frame.cols);
                int bottom = (int)(ssd_boxes[4 * i + 3] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;

				classIds.push_back((int)j - 1);;
				confidences.push_back((float)confidence);
				boxes.emplace_back(left, top, width, height);
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
