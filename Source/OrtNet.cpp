#include "OrtNet.h"

extern Ort::Env envOrtNet;

OrtNet::OrtNet() = default;
OrtNet::~OrtNet() = default;


void OrtNet::Init(const char* model_path) {

    // envOrtNet = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "OrtEnv");
    session_options.SetThreadPoolSize(4);
    session_options.SetGraphOptimizationLevel(2);
    session = Ort::Session(envOrtNet, model_path, session_options);

    // print number of model input nodes
    int num_input_nodes = session.GetInputCount();
    input_node_names = std::vector<const char*>(num_input_nodes);
    // std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        // printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_sizes.push_back(tensor_info.GetElementCount());

        // ONNXTensorElementDataType type = tensor_info.GetElementType();
        // printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims.push_back(tensor_info.GetShape());
        // printf("Input %d : num_dims=%zu\n", i, input_node_dims[i].size());
        // for (size_t j = 0; j < input_node_dims[i].size(); j++)
        //	printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[i][j]);
    }

    // print number of model output nodes
    int num_output_nodes = session.GetOutputCount();
    output_node_names = std::vector<const char*>(num_output_nodes);
    // std::cout << "Number of outputs = " << num_output_nodes << std::endl;

    // iterate over all input nodes
    for (int i = 0; i < num_output_nodes; i++) {
        // print output node names
        char* output_name = session.GetOutputName(i, allocator);
        // printf("Output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_node_sizes.push_back(tensor_info.GetElementCount());

        // ONNXTensorElementDataType type = tensor_info.GetElementType();
        // printf("Output %d : type=%d\n", i, type);

        // print output shapes/dims
        output_node_dims.push_back(tensor_info.GetShape());
        // printf("Output %d : num_dims=%zu\n", i, output_node_dims[i].size());
        // for (size_t j = 0; j < output_node_dims[i].size(); j++)
        //	printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[i][j]);
    }

}


void OrtNet::setInputTensor(const cv::Mat& frame)
{
	static cv::Mat blob;

	blob = cv::dnn::blobFromImage(
		frame,
		1.0,
		cv::Size(320,320),
		cv::Scalar(123, 117, 104),
		true, 
		false, 
		CV_32F);

	// auto bfi = blob.ptr<float[307200]>();
	// std::cout << bfi[] << std::endl;
	
	input_tensor = Ort::Value::CreateTensor<float>(
		allocator_info,
		blob.ptr<float>(),
		input_node_sizes[0],
		input_node_dims[0].data(),
		input_node_dims[0].size());
	assert(input_tensor.IsTensor());

/*
	auto it = input_tensor.GetTensorMutableData<float>();
	auto info = input_tensor.GetTensorTypeAndShapeInfo();
	auto cnt = info.GetElementCount();
	auto t = info.GetShape();
	cnt;
*/
}

void OrtNet::forward()
{
	output_tensor = session.Run(
		Ort::RunOptions{ nullptr },
		input_node_names.data(),
		&input_tensor,
		input_node_names.size(),
		output_node_names.data(),
		output_node_names.size());

	scores = output_tensor[0].GetTensorMutableData<float>();
	boxes = output_tensor[1].GetTensorMutableData<float>();
	outs = std::make_pair(scores, boxes);
}

std::pair<float*, float*> OrtNet::getOuts()
{
	return outs;
}

