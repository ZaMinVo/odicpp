# YOLO ONNX Inference with C++

Chạy mô hình YOLO (ONNX) bằng C++ sử dụng ONNX Runtime và OpenCV.

---

## 📦 Thư mục dự án
```
odicpp/
├── build/ result.output # Thư mục build
├── model/
│ └── yolo11n.onnx # Mô hình YOLO xuất sang ONNX
├── src/
│ ├── infer.cpp # File chính
│ ├── func.cpp # Tiền xử lý & hậu xử lý
│ └── func.hpp
├── test_data/
│ └── bus.jpg # Ảnh test
└── CMakeLists.txt
```
---

## 🚀 Hướng dẫn cài đặt

### 1. Cập nhật hệ thống

```
sudo apt update && sudo apt upgrade -y
```
### 2. Cài công cụ build và thư viện cần thiết
```
sudo apt install -y build-essential cmake git pkg-config g++ libopencv-dev
```
### 3. Cài Python & các gói Python (để export ONNX model)
```
sudo apt install -y python3-pip
pip install numpy onnx onnxruntime ultralytics
```
# Nếu bạn dùng CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# Nếu CUDA >= 12:
```
pip install torch
```
📥 Tải ONNX Runtime cho C++
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
```
Thư mục onnxruntime-linux-x64-1.18.0/ sẽ chứa thư viện và headers cho C++

🏗️ Build và chạy chương trình
```
mkdir build
cd build
cmake ..
make
./yolo
```
✅ Kết quả
Ảnh sau khi gắn box: result.jpg //chưa update

Kết quả đầu ra dạng JSON-like: result.output
