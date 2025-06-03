#ifndef FUNC_HPP
#define FUNC_HPP

#include <opencv2/opencv.hpp>
#include <iostream>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <chrono>

void saveDetections(const std::string& filename, const std::vector<std::vector<float>>& detections);

#endif