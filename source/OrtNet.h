#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <opencv2/dnn.hpp>


class OrtNet
{
public:
	OrtNet();
	~OrtNet();


#ifdef _WIN32
	void Init(const wchar_t* model_path);
#else
	void Init(const char* model_path);
#endif

	// Ort::Value getInputTensor(Mat blob);
	void setInputTensor(const cv::Mat& frame);
	void forward();
	std::pair<float(*)[1][3234][78], float(*)[1][3234][4]> getOuts();

private:
	// Ort Environment
	Ort::Env env = Ort::Env(nullptr);
	Ort::Session session = Ort::Session(nullptr);
	Ort::SessionOptions session_options;
	Ort::Allocator allocator = Ort::Allocator::CreateDefault();

	// Model ***
	// Inputs
	std::vector<const char*> input_node_names = std::vector<const char*>();
	std::vector<size_t> input_node_sizes = std::vector<size_t>();
	std::vector<std::vector<int64_t>> input_node_dims = std::vector<std::vector<int64_t>>();
	Ort::Value input_tensor = Ort::Value(nullptr);
	// Outputs
	std::vector<const char*>output_node_names = std::vector<const char*>();
	std::vector<size_t> output_node_sizes = std::vector<size_t>();
	std::vector<std::vector<int64_t>> output_node_dims = std::vector<std::vector<int64_t>>();
	std::vector<Ort::Value> output_tensor = std::vector<Ort::Value>();
	float(*scores)[1][3234][78] = NULL;
	float(*boxes)[1][3234][4] = NULL;
	std::pair<float(*)[1][3234][78], float(*)[1][3234][4]> outs = std::pair<float(*)[1][3234][78], float(*)[1][3234][4]>();
};
