import onnx
import onnxruntime as ort
import numpy as np


onnx_model = onnx.load("jetbot.onnx")

# Check that the model is well formed
onnx.checker.check_model(onnx_model)

ort_model = ort.InferenceSession("jetbot.onnx")

outputs = ort_model.run(
    None,
    {"obs": np.zeros((1, 74)).astype(np.float32)},
)
print(outputs)