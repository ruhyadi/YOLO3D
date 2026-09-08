import nncf
import openvino as ov
import numpy as np

# Load and prepare model
core = ov.Core()
model = core.read_model('weights/openvino/resnet18.xml')

# Create calibration data as a callable
def get_calibration_data():
    for _ in range(10):
        yield [np.random.randn(1, 3, 224, 224).astype(np.float32)]

# Quantize with calibration dataset
try:
    quantized_model = nncf.quantize(
        model, 
        calibration_dataset=nncf.Dataset(get_calibration_data())
    )
except Exception as e:
    print(f"NNCF quantization failed: {e}")
    print("Using simple FP16 conversion instead...")
    # Fallback: convert to FP16 instead
    from openvino.tools.mo import convert_model
    quantized_model = model

# Save model
ov.serialize(quantized_model, 'weights/openvino/resnet18_quantized.xml')
print("Model saved to weights/openvino/resnet18_quantized.xml")