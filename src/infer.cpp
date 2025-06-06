#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "func.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>



int main() {
    // Load model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, "../model/yolo11n.onnx", opts);

    // Load and preprocess image
    std::vector<float> inputTensor;
    cv::Mat img = loadAndPreprocess("../test_data/bus.jpg", inputTensor);

    // Input shape
    std::vector<int64_t> inputShape = {1, 3, 640, 640};

    // Create input tensor
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensorOrt = Ort::Value::CreateTensor<float>(
        memInfo, inputTensor.data(), inputTensor.size(),
        inputShape.data(), inputShape.size()
    );

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inputNamePtr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputNamePtr = session.GetOutputNameAllocated(0, allocator);

    const char* inputName = inputNamePtr.get();
    const char* outputName = outputNamePtr.get();

    std::vector<const char*> inputNames = {inputName};
    std::vector<const char*> outputNames = {outputName};

    // Run inference
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensorOrt, 1, outputNames.data(), 1);

    float* outputData = outputs[0].GetTensorMutableData<float>();
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // Post-processing (simple: lấy top N boxes)
    std::vector<std::vector<float>> boxes;
    size_t numDetections = outputShape[1]; // e.g., 8400
    size_t boxLen = outputShape[2];        // e.g., 6 = x, y, w, h, conf, class

    for (size_t i = 0; i < numDetections; ++i) {
        float conf = outputData[i * boxLen + 4];
        if (conf > 0.25f) {
            std::vector<float> box(6);
            for (int j = 0; j < 6; ++j)
                box[j] = outputData[i * boxLen + j];
            boxes.push_back(box);
        }
    }

    // Ghi output vào file .output dạng JSON-like
    std::ofstream out("result.output");
    out << std::fixed << std::setprecision(4);
    out << "[\n";

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        out << "  {\n"
            << "    \"x\": " << box[0] << ",\n"
            << "    \"y\": " << box[1] << ",\n"
            << "    \"w\": " << box[2] << ",\n"
            << "    \"h\": " << box[3] << ",\n"
            << "    \"conf\": " << box[4] << ",\n"
            << "    \"class\": " << static_cast<int>(box[5]) << "\n"
            << "  }";
        if (i != boxes.size() - 1) out << ",";
        out << "\n";
    }

    out << "]\n";
    out.close();
    std::cout << "Saved to result.output" << std::endl;


    // drawBoxes(img, boxes);
    // cv::imwrite("result.jpg", img);
    // std::cout << "Saved to result.jpg" << std::endl;
    return 0;
}
