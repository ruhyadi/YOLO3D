# Benchmarking Guide

This document explains how to use the provided scripts to benchmark inference performance for different execution modes: native PyTorch, PyTorch with `torch.compile` OpenVINO backend, ONNX models with OpenVINO Runtime, and OpenVINO IR models with OpenVINO Runtime.

All benchmark scripts are located in the `inference_scripts/` directory. They report average inference time in milliseconds (ms) and Frames Per Second (FPS). Higher FPS and lower ms indicate better performance.

**Prerequisites:**
- Ensure you have followed the setup steps in `README_openvino.md`, including environment setup, dependency installation, and model downloading/conversion.
- For benchmarks requiring specific model formats (e.g., ONNX, OpenVINO IR), make sure you have converted the models accordingly using the scripts in `weights/model_conversion/`, which save ONNX models to `weights/onnx/` and OpenVINO IR models to `weights/openvino/`.

## Full 3D Detection Pipeline Benchmark (`benchmark_full_pipeline.py`)

This script benchmarks the entire `detect3d` pipeline, which includes 2D object detection (YOLO-like) followed by 3D orientation and dimension regression for each detected object. It provides detailed timings for various stages within the pipeline.

- **Purpose:** To measure the end-to-end performance of the 3D object detection pipeline with different model backends (native PyTorch, PyTorch+`torch.compile` with OpenVINO, ONNX with OpenVINO Runtime, OpenVINO IR with OpenVINO Runtime).
- **Models Expected:**
    - A 2D detector model (PyTorch `.pt`, `.onnx`, or OpenVINO IR `.xml`).
    - A 3D regressor model (PyTorch `.pkl`, `.onnx`, or OpenVINO IR `.xml`).
- **Key Arguments:**
    - `--pipeline_type`: Specifies the backend for both models (e.g., `pytorch_native`, `onnx_openvino`).
    - `--yolo_weights`, `--yolo_cfg` (if applicable).
    - `--regressor_weights`, `--regressor_name` (e.g., 'resnet18').
    - `--dataset_path`: Path to input images (e.g., `eval/image_2/`).
    - `--calib_file`: Path to the camera calibration file.
    - `--img_size_yolo`, `--img_size_regressor`.
    - `--regressor_batch_size`: Batch size for processing detected 2D objects with the regressor.
    - `--device`: Target device (CPU, GPU, NPU).
- **Output:** Provides average times for the total pipeline and for individual components like YOLO preprocessing, YOLO inference, YOLO postprocessing, 3D object preprocessing, and 3D regressor inference.

**Example Command (Native PyTorch Pipeline):**
```bash
python inference_scripts/benchmark_full_pipeline.py \
  --pipeline_type pytorch_native \
  --yolo_weights weights/yolov5s.pt \
  --yolo_cfg models/yolov5s.yaml \
  --regressor_weights weights/resnet18.pkl \
  --regressor_name resnet18 \
  --dataset_path eval/image_2/ \
  --calib_file eval/camera_cal/calib_cam_to_cam.txt \
  --device cpu \
  --warmup_runs 3 \
  --benchmark_runs 10
```

**Example Command (OpenVINO IR Pipeline):**
```bash
python inference_scripts/benchmark_full_pipeline.py \
  --pipeline_type ir_openvino \
  --yolo_weights weights/openvino/yolov5s.xml \
  --regressor_weights weights/openvino/resnet18.xml \
  --regressor_name resnet18 \
  --dataset_path eval/image_2/ \
  --calib_file eval/camera_cal/calib_cam_to_cam.txt \
  --device CPU \
  --warmup_runs 3 \
  --benchmark_runs 10
```

**Example Command (OpenVINO ONNX Pipeline):**
```bash
python inference_scripts/benchmark_full_pipeline.py \
  --pipeline_type onnx_openvino \
  --yolo_weights weights/onnx/yolov5s.onnx \
  --yolo_cfg models/yolov5s.yaml \
  --regressor_weights weights/onnx/resnet18.onnx \
  --regressor_name resnet18 \
  --dataset_path eval/image_2/ \
  --calib_file eval/camera_cal/calib_cam_to_cam.txt \
  --device CPU \
  --warmup_runs 3 \
  --benchmark_runs 10
```
(Adapt paths and `--pipeline_type` for other configurations like `pytorch_compile_openvino`.)

### Benchmarking INT8 Quantized Models

Once the OpenVINO IR models have been quantized to INT8 (see [Quantize models to INT8](README_openvino.md#5-quantize-models-to-int8-optional) in `README_openvino.md`, producing `weights/openvino/yolov5s_int8.xml` and `weights/openvino/resnet18_quantized.xml`), benchmark them the same way as the regular OpenVINO IR pipeline above, just pointing `--yolo_weights`/`--regressor_weights` at the quantized files. The `--pipeline_type` stays `ir_openvino`.

**Example Command (INT8 OpenVINO IR Pipeline, CPU):**
```bash
python inference_scripts/benchmark_full_pipeline.py \
  --pipeline_type ir_openvino \
  --yolo_weights weights/openvino/yolov5s_int8.xml \
  --regressor_weights weights/openvino/resnet18_quantized.xml \
  --regressor_name resnet18 \
  --dataset_path eval/image_2/ \
  --calib_file eval/camera_cal/calib_cam_to_cam.txt \
  --device CPU \
  --warmup_runs 3 \
  --benchmark_runs 10
```
Swap `--device CPU` for `GPU` or `NPU` to benchmark on other targets, exactly like the FP32 IR pipeline. Results are comparable directly against the FP32 IR run since both use the same `ir_openvino` pipeline type and inputs — only the model weights differ.

## Individual Model Benchmark Scripts

### 1. Native PyTorch (`benchmark_pytorch_native.py`)
   - **Purpose:** Measures the inference speed of a PyTorch model running natively on the specified device (CPU or CUDA) without any OpenVINO optimizations.
   - **Model Expected:** PyTorch model (`.pt` file) and its configuration (`.yaml` file).
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_pytorch_native.py \
       --weights weights/yolov5s.pt \
       --cfg models/yolov5s.yaml \
       --img_size 640 \
       --batch_size 1 \
       --device cpu \
       --warmup_runs 10 \
       --benchmark_runs 50 \
       --dataset_path eval/image_2/ # Optional: Path to dataset for real data benchmarking
     ```
     (Replace `cpu` with `cuda` if you have a compatible GPU and PyTorch GPU version installed.)

### 2. PyTorch with `torch.compile` OpenVINO Backend (`benchmark_torch.py`)
   - **Purpose:** Measures inference speed when using `torch.compile` with the OpenVINO backend. This allows PyTorch models to be accelerated by OpenVINO with minimal code changes.
   - **Model Expected:** PyTorch model (`.pt` file) and its configuration (`.yaml` file).
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_torch.py \
       --weights weights/yolov5s.pt \
       --cfg models/yolov5s.yaml \
       --img_size 640 \
       --batch_size 1 \
       --device CPU \ # OpenVINO device: CPU, GPU, etc.
       --warmup_runs 10 \
       --benchmark_runs 50 \
       --dataset_path eval/image_2/ # Optional: Path to dataset for real data benchmarking
     ```

### 3. ONNX with OpenVINO Runtime (`benchmark_onnx.py`)
   - **Purpose:** Measures inference speed of an ONNX model using the OpenVINO Runtime.
   - **Model Expected:** ONNX model (`.onnx` file). Ensure this model is converted from PyTorch or other formats first (e.g., using `pytorch_to_openvino_ir.py` which saves the intermediate ONNX to `weights/onnx/`).
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_onnx.py \
       --model_path weights/onnx/yolov5s.onnx \
       --img_size 640 \
       --batch_size 1 \
       --device CPU \ # OpenVINO device
       --warmup_runs 10 \
       --benchmark_runs 50 \
       --dataset_path eval/image_2/ # Optional: Path to dataset for real data benchmarking
     ```

### 4. OpenVINO IR with OpenVINO Runtime (`benchmark_openvino_ir.py`)
   - **Purpose:** Measures inference speed of an OpenVINO Intermediate Representation (IR) model (`.xml` and `.bin` files) using the OpenVINO Runtime. This is often the most optimized path for OpenVINO.
   - **Model Expected:** OpenVINO IR model (`.xml` file). Ensure this model is converted first (e.g., using `pytorch_to_openvino_ir.py` which saves IR models to `weights/openvino/`).
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_openvino_ir.py \
       --model_path weights/openvino/yolov5s.xml \
       --img_size 640 \
       --batch_size 1 \
       --device CPU \
       --warmup_runs 10 \
       --benchmark_runs 50 \
       --dataset_path eval/image_2/ # Optional: Path to dataset for real data benchmarking
     ```



## Notes:
- **Device:** Performance will vary significantly based on the `--device` used (CPU, GPU, NPU) and the hardware capabilities.
- **First Run:** The first time you run an OpenVINO benchmark (especially with `torch.compile` or for a new model/device combination), there might be a one-time model compilation/optimization cost. Subsequent runs will be faster. The warm-up runs in the scripts help mitigate this in reported averages.
- **Benchmarking with Real Data vs. Dummy Data:** Using the `--dataset_path` option to benchmark with real images can provide more realistic performance figures as it accounts for actual data characteristics and I/O, though the current implementation pre-loads all images to minimize I/O variability during timing loops. Dummy data is useful for quick checks and isolating pure computation speed.


