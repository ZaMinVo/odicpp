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
pip install onnx
# Cuda > 12
pip install torch 
# Cuda 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt g++
sudo apt install -y g++


wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz

<!-- wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -P src/ -->

