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

inline void preprocess(const Mat& frame, Size inpSize, float scale, const Scalar& mean, bool swapRB);

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

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

/*
	// initialize  enviroment...one enviroment per process
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetThreadPoolSize(1);
	session_options.SetGraphOptimizationLevel(2);

#ifdef _WIN32
	const wchar_t* model_path = L"D:/REPOs/ML/ssdIE/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx";
#else
	const char* model_path = "D:/REPOs/ML/ssdIE/dnnOCV/build/RelWithDebInfo/model.onnx";
#endif

	std::cerr << "Using Onnxruntime C++ API" << '\n';
	Ort::Session session(env, model_path, session_options);

	Ort::Allocator allocator = Ort::Allocator::CreateDefault();

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;
	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto elements = tensor_info.GetElementCount();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}

	// print number of model output nodes
	size_t num_output_nodes = session.GetOutputCount();
	std::vector<const char*> output_node_names(num_output_nodes);
	std::vector<int64_t> output_node_dims;
	printf("Number of outputs = %zu\n", num_output_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_output_nodes; i++) {
		// print output node names
		char* output_name = session.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);
		output_node_names[i] = output_name;

		// print output node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		// print output shapes/dims
		output_node_dims = tensor_info.GetShape();
		printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
		for (int j = 0; j < output_node_dims.size(); j++)
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
	}


  //*************************************************************************
  // Score the model using sample data, and inspect values

	size_t input_tensor_size = 1 * 3 * 320 * 320;	// simplify ... using known dim values to calculate size
													// use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values(input_tensor_size);
	// std::vector<const char*> output_node_names = { "boxes", "scores" };

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float)i / (input_tensor_size + 1);

	// create input tensor object from data values
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	// for (int i = 0; i < 10; i++)
	// {
		auto start = std::chrono::high_resolution_clock::now();
		auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
		auto end = std::chrono::high_resolution_clock::now();
		assert(output_tensors.size() == 2 && output_tensors.front().IsTensor());

		auto duration = duration_cast<milliseconds>(end - start);
		std::cout << "inference taken : " << duration.count() << " ms" << "\n";
	// }
	

	// Get pointer to output tensor float values
	auto scores = output_tensors[0].GetTensorMutableData<float[1][3234][78]>();
	auto boxes = output_tensors[1].GetTensorMutableData<float[1][3234][4]>();
	
	// score the model, and print scores for first 5 classes
	// for (int i = 0; i < 5; i++)
	//	printf("Score for class [%d] =  %f\n", i, scores[i]);

	// Results should be as below...
	// Score for class[0] = 0.000045
	// Score for class[1] = 0.003846
	// Score for class[2] = 0.000125
	// Score for class[3] = 0.001180
	// Score for class[4] = 0.001317

	printf("Done!\n");

*/

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	int initialConf = (int)(confThreshold * 100);
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	if (parser.has("input"))
		cap.open(parser.get<String>("input"));
	else
		cap.open(parser.get<int>("device"));

#ifdef CV_CXX11
	bool process = true;

	// Frames capturing thread
	QueueFPS<Mat> framesQueue;
	std::thread framesThread([&]() {
		Mat frame;
		while (process)
		{
			cap >> frame;
			if (!frame.empty())
				framesQueue.push(frame.clone());
			else
				break;
		}
		});

	// Frames processing thread
	QueueFPS<Mat> processedFramesQueue;
	QueueFPS<std::vector<Mat> > predictionsQueue;
	std::thread processingThread([&]() {
		std::queue<AsyncArray> futureOutputs;
		Mat blob;
		while (process)
		{
			// Get a next frame
			Mat frame;
			{
				if (!framesQueue.empty())
				{
					frame = framesQueue.get();
					if (async)
					{
						if (futureOutputs.size() == async)
							frame = Mat();
					}
					else
						framesQueue.clear();  // Skip the rest of frames
				}
			}

			// Process the frame
			if (!frame.empty())
			{
				preprocess(frame, Size(320, 320), 1.0, Scalar(123, 117, 104), true);
				processedFramesQueue.push(frame);

				/*
				if (async)
				{
					futureOutputs.push(net.forwardAsync());
				}
				else
				{
					std::vector<Mat> outs;
					net.forward(outs, outNames);
					predictionsQueue.push(outs);
				}
				*/
			}

			while (!futureOutputs.empty() &&
				futureOutputs.front().wait_for(std::chrono::seconds(0)))
			{
				AsyncArray async_out = futureOutputs.front();
				futureOutputs.pop();
				Mat out;
				async_out.get(out);
				predictionsQueue.push({ out });
			}
		}
		});

	// Postprocessing and rendering loop
	while (waitKey(1) < 0)
	{
		/*
		if (predictionsQueue.empty())
			continue;
		*/

		// std::vector<Mat> outs = predictionsQueue.get();
		if (processedFramesQueue.empty())
			continue;

		Mat frame = processedFramesQueue.get();

		// postprocess(frame, outs, net);

		// if (predictionsQueue.counter > 1)
		{
			std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
			putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

			// label = format("Network: %.2f FPS", predictionsQueue.getFPS());
			// putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

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

inline void preprocess(const Mat& frame, Size inpSize, float scale, const Scalar& mean, bool swapRB)
{
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	static Mat blob = blobFromImage(frame, 1.0, inpSize, mean, swapRB, false, CV_32F);
	auto s = blob.size.p;
	// std::cout << s[0] << " - " << s[1] << " - " << s[2] << " - " << s[3] << "\n";
	// create input tensor object from data values
	int64_t input_tensor_shape[4] = {s[0], s[1], s[2], s[2]};
	size_t input_tensor_size = s[0] * s[1] * s[2] * s[2];
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>( allocator_info, blob.ptr<float>(), 
		input_tensor_size, input_tensor_shape, blob.dims);
	assert(input_tensor.IsTensor());
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;
	if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() > 0);
		for (size_t k = 0; k < outs.size(); k++)
		{
			float* data = (float*)outs[k].data;
			for (size_t i = 0; i < outs[k].total(); i += 7)
			{
				float confidence = data[i + 2];
				if (confidence > confThreshold)
				{
					int left = (int)data[i + 3];
					int top = (int)data[i + 4];
					int right = (int)data[i + 5];
					int bottom = (int)data[i + 6];
					int width = right - left + 1;
					int height = bottom - top + 1;
					if (width * height <= 1)
					{
						left = (int)(data[i + 3] * frame.cols);
						top = (int)(data[i + 4] * frame.rows);
						right = (int)(data[i + 5] * frame.cols);
						bottom = (int)(data[i + 6] * frame.rows);
						width = right - left + 1;
						height = bottom - top + 1;
					}
					classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
					boxes.push_back(Rect(left, top, width, height));
					confidences.push_back(confidence);
				}
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

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
