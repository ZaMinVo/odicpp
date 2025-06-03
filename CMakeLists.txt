cmake_minimum_required(VERSION 3.10)
project(YOLO_TRT_Live)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    /usr/include
    /usr/local/cuda/include
    /usr/include/x86_64-linux-gnu
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)

add_executable(infer src/infer.cpp)

target_link_libraries(infer nvinfer cudart ${OpenCV_LIBS})
