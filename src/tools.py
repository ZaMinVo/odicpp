from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import shutil
import torch
import os

def prep_model(model_path = "model/yolo11n.pt"):
    model = YOLO(model_path)  
    return model

def prep_data():
    model = prep_model()
    results = model("https://ultralytics.com/images/bus.jpg") 
    shutil.move("bus.jpg", "test_data/bus.jpg")
    return results

def convert_to_onnx(model_path = "model/yolo11n.pt", output_path = "model/yolo11n.onnx", input_shape = (640, 640)):
    
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=input_shape, dynamic=True, simplify=True, opset=11)

    default_onnx = os.path.splitext(model_path)[0] + ".onnx"
    if default_onnx != output_path and os.path.exists(default_onnx):
        os.rename(default_onnx, output_path)

def test_onnx(onnx_path="model/yolo11n.onnx", img_size=(640, 640)):
    session = ort.InferenceSession(onnx_path)
    inputs = session.get_inputs()
    input_shape = []
    for s in inputs[0].shape:
        if isinstance(s, str) or s is None:
            # Replace dynamic axes with default values
            if s == 'batch' or s is None:
                input_shape.append(1)
            elif s == 'height':
                input_shape.append(img_size[0])
            elif s == 'width':
                input_shape.append(img_size[1])
            else:
                raise ValueError(f"Unknown dynamic axis: {s}")
        else:
            input_shape.append(s)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    result = session.run(None, {inputs[0].name: dummy_input})
    print("Inference successful! Output shapes:")
    for out, val in zip(session.get_outputs(), result):
        print(f"  {out.name}: {val.shape}")
