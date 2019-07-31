#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

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
	// Outputs
	std::vector<const char*>output_node_names = std::vector<const char*>();
	std::vector<size_t> output_node_sizes = std::vector<size_t>();
	std::vector<std::vector<int64_t>> output_node_dims = std::vector<std::vector<int64_t>>();
};
