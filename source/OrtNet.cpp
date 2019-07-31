#include "OrtNet.h"

OrtNet::OrtNet()
{
}

OrtNet::~OrtNet()
{
}

#ifdef _WIN32
void OrtNet::Init(const wchar_t* model_path)
{
	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ssdOCV");
	session_options.SetThreadPoolSize(1);
	session_options.SetGraphOptimizationLevel(2);

	std::cout << "Using Onnxruntime C++ API" << std::endl;
	session = Ort::Session(env, model_path, session_options);

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	input_node_names = std::vector<const char*>(num_input_nodes);
	std::cout << "Number of inputs = " << num_input_nodes << std::endl;

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		input_node_sizes.push_back(tensor_info.GetElementCount());

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims.push_back(tensor_info.GetShape());
		printf("Input %d : num_dims=%zu\n", i, input_node_dims[i].size());
		for (int j = 0; j < input_node_dims[i].size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[i][j]);
	}

	// print number of model output nodes
	size_t num_output_nodes = session.GetOutputCount();
	output_node_names = std::vector<const char*>(num_output_nodes);
	std::cout << "Number of outputs = " << num_output_nodes << std::endl;

	// iterate over all input nodes
	for (int i = 0; i < num_output_nodes; i++) {
		// print output node names
		char* output_name = session.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);
		output_node_names[i] = output_name;

		// print output node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		output_node_sizes.push_back(tensor_info.GetElementCount());

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		// print output shapes/dims
		output_node_dims.push_back(tensor_info.GetShape());
		printf("Output %d : num_dims=%zu\n", i, output_node_dims[i].size());
		for (int j = 0; j < output_node_dims[i].size(); j++)
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[i][j]);
	}


	//*************************************************************************
	// Score the model using sample data, and inspect values

	std::vector<float> input_tensor_values(input_node_sizes[0]);
	// std::vector<const char*> output_node_names = { "boxes", "scores" };

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_node_sizes[0]; i++)
		input_tensor_values[i] = (float)i / (input_node_sizes[0] + 1);

	// create input tensor object from data values
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_node_sizes[0], input_node_dims[0].data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	// for (int i = 0; i < 10; i++)
	// {
	auto start = std::chrono::high_resolution_clock::now();
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
	auto end = std::chrono::high_resolution_clock::now();
	assert(output_tensors.size() == 2 && output_tensors.front().IsTensor());

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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


}
#else
void OrtNet::Init(const char* model_path)
{

}
#endif

