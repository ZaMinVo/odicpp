# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài công cụ build và compiler
sudo apt install -y build-essential cmake git pkg-config

# Cài đặt OpenCV
sudo apt install -y libopencv-dev

# Cài đặt pip
sudo apt install python3-pip

# Cài đặt numpy
pip install numpy

# Cài đặt ultralytics
pip install ultralytics

# Cài đặt ONNX Runtime cho Python (để export model)
pip install onnxruntime

# Cài đặt torch và onnx
pip install torch onnx

# Cài đặt g++
sudo apt install -y g++

# Cài đặt TensorRT
pip install tensorrt

# Chuyển đổi ONNX sang TensorRT
trtexec --onnx=model/yolo11n.onnx --saveEngine=model/yolo11n_fp16.engine --fp16

compile: g++ -Wall -o testcv src/testcv.cpp `pkg-config --cflags --libs opencv4`