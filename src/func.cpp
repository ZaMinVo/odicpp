#include "func.hpp"

cv::Mat loadAndPreprocess(const std::string& imagePath, std::vector<float>& inputTensor) {
    cv::Mat img = cv::imread(imagePath);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));  // input size

    // Normalize [0,255] → [0,1]
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    // HWC to CHW
    inputTensor.resize(3 * 640 * 640);
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 640; ++y)
            for (int x = 0; x < 640; ++x)
                inputTensor[idx++] = resized.at<cv::Vec3f>(y, x)[c];

    return img;
}

void drawBoxes(cv::Mat& image, const std::vector<std::vector<float>>& boxes) {
    for (const auto& box : boxes) {
        float x = box[0], y = box[1], w = box[2], h = box[3], conf = box[4];
        int left = static_cast<int>(x - w / 2);
        int top = static_cast<int>(y - h / 2);
        int right = static_cast<int>(x + w / 2);
        int bottom = static_cast<int>(y + h / 2);

        if (conf > 0.25f) {
            cv::rectangle(image, cv::Rect(left, top, right - left, bottom - top),
                          cv::Scalar(0, 255, 0), 2);
        }
    }
}
