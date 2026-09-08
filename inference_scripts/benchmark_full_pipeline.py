# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import numpy as np
import torch
from pathlib import Path
import sys
import collections # For defaultdict

# Add project root for custom module imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # Assuming this script is in inference_scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from utils.datasets import LoadImages
    from inference_scripts.benchmarkable_detect3d_pipeline import benchmarkable_detect3d, load_yolo_detector, load_regressor_model
    from script.Dataset import ClassAverages, generate_bins # Assuming ClassAverages is in script.Dataset
    # from utils.torch_utils import select_device # Using torch.device directly
except ImportError as e:
    print(f"Error: Failed to import necessary modules: {e}. Ensure PYTHONPATH is correctly set.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Full 3D Detection Pipeline Benchmark Script")
    parser.add_argument('--pipeline_type', type=str, required=True,
                        choices=['pytorch_native', 'pytorch_compile_openvino', 'onnx_openvino', 'ir_openvino'],
                        help="Type of pipeline to benchmark.")
    parser.add_argument('--yolo_weights', type=str, required=True, help="Path to 2D detector model file (e.g., .pt, .onnx, .xml).")
    parser.add_argument('--yolo_cfg', type=str, help="Path to YOLO model configuration YAML (required for PyTorch types).")
    parser.add_argument('--regressor_weights', type=str, required=True, help="Path to 3D regressor model file (e.g., .pkl, .onnx, .xml).")
    parser.add_argument('--regressor_name', type=str, default='resnet18', help="Name of the regressor model (e.g., 'resnet18').")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to input image dataset directory or single image file.")
    parser.add_argument('--calib_file', type=str, required=True, help="Path to camera calibration file.")

    parser.add_argument('--img_size_yolo', type=int, default=640, help="Target image size (height=width) for YOLO detector input.")
    parser.add_argument('--img_size_regressor', type=int, default=224, help="Target image size (height=width) for Regressor input patches.")

    # Note: Full pipeline processes one image at a time. YOLO batch size is effectively 1 here.
    # Regressor batch size is for processing multiple detected objects from a single image.
    parser.add_argument('--regressor_batch_size', type=int, default=16, help="Batch size for processing detected objects with the regressor.")

    parser.add_argument('--warmup_runs', type=int, default=5, help="Number of warm-up runs through the dataset (or part of it).")
    parser.add_argument('--benchmark_runs', type=int, default=20, help="Number of benchmark runs through the dataset (or part of it).")
    parser.add_argument('--device', type=str, default='cpu', help="Device: 'cpu' or 'cuda' for PyTorch native; 'CPU', 'GPU', etc. for OpenVINO backends/runtimes.")

    parser.add_argument('--conf_thres_yolo', type=float, default=0.25, help="Confidence threshold for YOLO NMS.")
    parser.add_argument('--iou_thres_yolo', type=float, default=0.45, help="IOU threshold for YOLO NMS.")
    parser.add_argument('--yolo_classes_to_detect', type=int, nargs='+', default=None, help="Filter by class: e.g., 0, or 0 2 3. Default: all classes.")
    parser.add_argument('--max_detections_yolo', type=int, default=1000, help="Maximum number of detections per image from YOLO.")

    args = parser.parse_args()

    # Validate conditional cfg path
    if args.pipeline_type in ['pytorch_native', 'pytorch_compile_openvino'] and not args.yolo_cfg:
        print("Error: --yolo_cfg is required for PyTorch-based pipeline types.", file=sys.stderr)
        sys.exit(1)

    # --- Device Setup ---
    # For PyTorch native, device_str is 'cpu' or 'cuda'.
    # For torch.compile OpenVINO, device_str is OpenVINO device type ('CPU', 'GPU').
    # For OpenVINO ONNX/IR, device_str is OpenVINO device type ('CPU', 'GPU').
    pytorch_target_device_str = args.device if args.pipeline_type == 'pytorch_native' else 'cpu'
    pytorch_device = torch.device(pytorch_target_device_str)
    openvino_target_device_str = args.device.upper() # OpenVINO expects uppercase device names

    # --- Load Models ---
    print(f"--- Loading models for pipeline type: {args.pipeline_type} ---")
    yolo_model = load_yolo_detector(
        args.yolo_weights, args.yolo_cfg,
        device_str=args.device, # Pass the original device string
        model_type=args.pipeline_type,
        img_size=(args.img_size_yolo, args.img_size_yolo), # Pass as tuple
        batch_size=1 # YOLO processes one image at a time in this pipeline
    )
    if yolo_model is None: sys.exit("YOLO model loading failed.")

    regressor_model = load_regressor_model(
        args.regressor_weights, args.regressor_name,
        device_str=args.device, # Pass the original device string
        model_type=args.pipeline_type,
        img_size=(args.img_size_regressor, args.img_size_regressor), # Pass as tuple
        batch_size=args.regressor_batch_size # Regressor can batch internal detections
    )
    if regressor_model is None: sys.exit("Regressor model loading failed.")

    # --- Load Utilities ---
    try:
        class_averages = ClassAverages() # Assumes 'class_dims.txt' is in working dir or findable
        angle_bins = generate_bins(2) # Example: 2 bins for orientation
    except Exception as e:
        print(f"Error loading ClassAverages or generating angle_bins: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Determine YOLO Stride and Class Names ---
    yolo_model_stride = 32
    yolo_class_names = [f'class_{i}' for i in range(80)] # Default
    if args.pipeline_type in ['pytorch_native', 'pytorch_compile_openvino']:
        yolo_eval_model = yolo_model._model if hasattr(yolo_model, '_model') else yolo_model # For compiled model
        if hasattr(yolo_eval_model, 'stride'):
            yolo_model_stride = int(yolo_eval_model.stride.max().item() if isinstance(yolo_eval_model.stride, torch.Tensor) else yolo_eval_model.stride)
        if hasattr(yolo_eval_model, 'names'):
            yolo_class_names = yolo_eval_model.names
        print(f"Using YOLO Stride: {yolo_model_stride}, Class Names: {yolo_class_names[:3]}...")
    else: # For OpenVINO models, stride might need to be known or passed as arg. Using default.
        print(f"Using default YOLO Stride: {yolo_model_stride} for ONNX/IR. Class names are placeholders.")


    # --- Load Dataset ---
    print(f"Loading dataset from: {args.dataset_path}...")
    # LoadImages expects img_size as int for square, stride, auto=False for consistent padding
    dataset_loader = LoadImages(args.dataset_path, img_size=args.img_size_yolo, stride=yolo_model_stride, auto=False)
    all_images_hwc_bgr_np = []
    for path, img_letterboxed_chw_rgb_np, img_original_hwc_bgr, _, _ in dataset_loader:
        # We need the original HWC BGR image for DetectedObject
        all_images_hwc_bgr_np.append(img_original_hwc_bgr)

    if not all_images_hwc_bgr_np:
        print(f"Error: No images found at {args.dataset_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(all_images_hwc_bgr_np)} images for benchmarking.")

    # --- Warm-up Phase ---
    for i in range(args.warmup_runs):
        image_np_hwc = all_images_hwc_bgr_np[i % len(all_images_hwc_bgr_np)]
        _, _ = benchmarkable_detect3d(
            yolo_model=yolo_model, regressor_model=regressor_model,
            yolo_model_type=args.pipeline_type, regressor_model_type=args.pipeline_type,
            input_image_np=image_np_hwc,
            img_size_yolo=(args.img_size_yolo, args.img_size_yolo), stride_yolo=yolo_model_stride,
            img_size_regressor=(args.img_size_regressor, args.img_size_regressor),
            device_pytorch=pytorch_device, device_openvino=openvino_target_device_str,
            calib_file_path=args.calib_file, class_averages=class_averages, angle_bins=angle_bins,
            conf_thres_yolo=args.conf_thres_yolo, iou_thres_yolo=args.iou_thres_yolo,
            yolo_classes_to_detect=args.yolo_classes_to_detect, max_detections_yolo=args.max_detections_yolo,
            regressor_batch_size=args.regressor_batch_size
        )

    # --- Benchmark Phase ---
    overall_timings_list = []
    for i in range(args.benchmark_runs):
        image_np_hwc = all_images_hwc_bgr_np[i % len(all_images_hwc_bgr_np)]
        _, frame_timings = benchmarkable_detect3d(
            yolo_model=yolo_model, regressor_model=regressor_model,
            yolo_model_type=args.pipeline_type, regressor_model_type=args.pipeline_type,
            input_image_np=image_np_hwc,
            img_size_yolo=(args.img_size_yolo, args.img_size_yolo), stride_yolo=yolo_model_stride,
            img_size_regressor=(args.img_size_regressor, args.img_size_regressor),
            device_pytorch=pytorch_device, device_openvino=openvino_target_device_str,
            calib_file_path=args.calib_file, class_averages=class_averages, angle_bins=angle_bins,
            conf_thres_yolo=args.conf_thres_yolo, iou_thres_yolo=args.iou_thres_yolo,
            yolo_classes_to_detect=args.yolo_classes_to_detect, max_detections_yolo=args.max_detections_yolo,
            regressor_batch_size=args.regressor_batch_size
        )
        overall_timings_list.append(frame_timings)

    # --- Results Aggregation and Output ---
    if not overall_timings_list:
        print("No benchmark timings recorded.", file=sys.stderr)
        sys.exit(1)

    aggregated_timings = collections.defaultdict(list)
    for frame_time_dict in overall_timings_list:
        for key, value in frame_time_dict.items():
            aggregated_timings[key].append(value)

    avg_timings_ms = {k: np.mean(v) * 1000 for k, v in aggregated_timings.items()}

    total_pipeline_time_ms = sum(avg_timings_ms.values())
    # Exclude num_regressor_runs from sum as it's a count, not time
    if 'num_regressor_runs' in avg_timings_ms:
        total_pipeline_time_ms -= avg_timings_ms['num_regressor_runs']


    overall_fps = 1000.0 / total_pipeline_time_ms if total_pipeline_time_ms > 0 else 0

    print("\n--- Full 3D Detection Pipeline Benchmark Results ---")
    print(f"Pipeline Type: {args.pipeline_type}")
    print(f"YOLO Model: {args.yolo_weights}")
    if args.yolo_cfg: print(f"YOLO Config: {args.yolo_cfg}")
    print(f"Regressor Model: {args.regressor_weights} (Type: {args.regressor_name})")
    print(f"Dataset: {args.dataset_path} ({len(all_images_hwc_bgr_np)} images used for cycling)")
    if args.pipeline_type == 'pytorch_native':
        print(f"PyTorch Native Device: {pytorch_device.type}")
    elif args.pipeline_type in ['onnx_openvino', 'ir_openvino', 'pytorch_compile_openvino']:
        print(f"Target OpenVINO Device: {openvino_target_device_str}")
    print(f"YOLO Input Size: {args.img_size_yolo}x{args.img_size_yolo}")
    print(f"Regressor Input Patch Size: {args.img_size_regressor}x{args.img_size_regressor}")
    print(f"Regressor Batch Size (for detected objects): {args.regressor_batch_size}")
    print(f"Warm-up Runs: {args.warmup_runs}, Benchmark Runs: {args.benchmark_runs}")
    print("----------------------------------------------------")
    print("Average Time per Component (ms):")
    for key, avg_ms in avg_timings_ms.items():
        if key == 'num_regressor_runs':
            print(f"  - Avg Num Regressor Runs per Image: {avg_ms/1000:.2f}") # Convert back from ms for count
        else:
            print(f"  - {key.replace('_', ' ').capitalize()}: {avg_ms:.3f} ms")
    print("----------------------------------------------------")
    print(f"Total Average Pipeline Time: {total_pipeline_time_ms:.3f} ms")
    print(f"Overall Pipeline FPS: {overall_fps:.2f}")
    print("----------------------------------------------------\n")

if __name__ == '__main__':
    main()
