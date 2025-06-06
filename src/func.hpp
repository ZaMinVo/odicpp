#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Load ảnh và resize
cv::Mat loadAndPreprocess(const std::string& imagePath, std::vector<float>& inputTensor);

// Vẽ bounding boxes
void drawBoxes(cv::Mat& image, const std::vector<std::vector<float>>& boxes);
