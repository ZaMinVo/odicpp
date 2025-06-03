#include 'func.hpp'

std::string enginePath = "../model/yolo11n.engine"

using namespace nvinfer1;

int main() {
    std::string engine_path = "../model/yolo11n_fp16.engine";
    std::vector<char> engine_data = loadEngine(engine_path);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    IExecutionContext* context = engine->createExecutionContext();

    // Load camera
    cv::VideoCapture cap(0); // Or a video path
    if (!cap.isOpened()) return -1;

    int inputIndex = engine->getBindingIndex("images"); 
    int outputIndex = engine->getBindingIndex("output"); 

    const int input_w = 640, input_h = 640;
    const int input_size = 3 * input_w * input_h;
    const int output_size = 1000;

    float* input_host = new float[input_size];
    float* output_host = new float[output_size];

    void *input_dev, *output_dev;
    cudaMalloc(&input_dev, input_size * sizeof(float));
    cudaMalloc(&output_dev, output_size * sizeof(float));

    void* bindings[2] = {input_dev, output_dev};

    int frame_id = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_w, input_h));
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        // HWC to CHW
        std::vector<cv::Mat> chw(3);
        for (int i = 0; i < 3; ++i)
            chw[i] = cv::Mat(input_h, input_w, CV_32F, input_host + i * input_w * input_h);
        cv::split(resized, chw);

        cudaMemcpy(input_dev, input_host, input_size * sizeof(float), cudaMemcpyHostToDevice);
        context->enqueueV2(bindings, 0, nullptr);
        cudaMemcpy(output_host, output_dev, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<std::vector<float>> detections;
        for (int i = 0; i < output_size; i += 6) {
            if (output_host[i + 4] > 0.3f)  // confidence threshold
                detections.push_back({
                    output_host[i], output_host[i+1], output_host[i+2],
                    output_host[i+3], output_host[i+4], output_host[i+5]
                });
        }

        std::string out_name = "../output/frame_" + std::to_string(frame_id++) + ".txt";
        saveDetections(out_name, detections);

        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 27) break; //ESC
    }

    cap.release();
    context->destroy(); engine->destroy(); runtime->destroy();
    cudaFree(input_dev); cudaFree(output_dev);
    delete[] input_host; delete[] output_host;

    return 0;
}