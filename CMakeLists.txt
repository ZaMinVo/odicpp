cmake_minimum_required(VERSION 3.10)
project(YOLO_ONNX_CPP)

set(CMAKE_CXX_STANDARD 14)

# ONNX Runtime paths
include_directories(/home/jet/onnxruntime-linux-x64-1.18.0/include)
link_directories(/home/jet/onnxruntime-linux-x64-1.18.0/lib)

# OpenCV
find_package(OpenCV REQUIRED)

# Build binary
add_executable(yolo src/func.cpp src/infer.cpp)

# Link libraries
target_link_libraries(yolo ${OpenCV_LIBS} onnxruntime)