# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import numpy
import time
import argparse
from pathlib import Path
import sys
import cv2 # Explicitly import cv2
import random # For selecting images

# Add project root to sys.path for utils.datasets
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from utils.datasets import LoadImages # For loading real datasets
except ImportError as e:
    print(f"Error: Could not import 'LoadImages' from 'utils.datasets'. {e}", file=sys.stderr)
    # Dummy class if import fails
    class LoadImages:
        def __init__(self, p, img_size=640, stride=32, auto=True, vid_stride=1): raise ImportError("Real LoadImages not available.")
        def __iter__(self): return self
        def __next__(self): raise StopIteration
        def __len__(self): return 0


def main():
    parser = argparse.ArgumentParser(description="ONNX Model + OpenVINO Runtime Benchmark Script with Real Dataset")
    parser.add_argument('--model_path', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset directory or image file. If None, uses dummy data.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size (square) for preprocessing and model input. Must match ONNX model expected input size.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference. Model must support this or have dynamic batch.')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warm-up inference runs')
    parser.add_argument('--benchmark_runs', type=int, default=50, help='Number of benchmark inference runs')
    parser.add_argument('--device', type=str, default='CPU', help="OpenVINO device (e.g., 'CPU', 'GPU').")

    args = parser.parse_args()

    onnx_model_path = Path(args.model_path)
    if not onnx_model_path.exists():
        print(f"Error: ONNX model file not found: {args.model_path}", file=sys.stderr)
        return
    if args.dataset_path and not Path(args.dataset_path).exists() and not Path(args.dataset_path).is_file():
        if not (Path(args.dataset_path).parent.exists() and '*' in Path(args.dataset_path).name):
             print(f"Error: Dataset path {args.dataset_path} does not exist or is not a file/glob pattern.", file=sys.stderr)
             return

    # OpenVINO Model Loading from ONNX
    final_input_layer = None # To be determined after compilation
    final_input_shape_tuple = None
    element_type_numpy = numpy.float32 # Default, will be updated from model
    compiled_model = None # Initialize

    try:
        print("Initializing OpenVINO Core...")
        core = ov.Core()
        print(f"Reading ONNX model from: {args.model_path}")
        model = core.read_model(model=args.model_path)

        if len(model.inputs) != 1:
            print(f"Warning: Model has {len(model.inputs)} inputs. This script primarily supports single input models for dataset processing.", file=sys.stderr)

        input_tensor_pre_reshape = model.input(0)
        desired_input_shape_list = [args.batch_size, 3, args.img_size, args.img_size]

        try:
            reshape_dict = {input_tensor_pre_reshape.any_name: ov.PartialShape(desired_input_shape_list)}
            model.reshape(reshape_dict)
            print(f"Attempted to reshape model to: {desired_input_shape_list} before compilation.")
        except Exception as e:
            print(f"Warning: Could not reshape ONNX model to {desired_input_shape_list}. Using model's existing shape. Error: {e}", file=sys.stderr)
            current_partial_shape = model.input(0).get_partial_shape()
            if current_partial_shape[0].is_static and current_partial_shape[0].get_length() != args.batch_size:
                print(f"Error: Model expects batch size {current_partial_shape[0].get_length()} but {args.batch_size} was provided, and reshape failed.", file=sys.stderr)
                return

        print(f"Compiling model for device: {args.device.upper()}...")
        compiled_model = core.compile_model(model, device_name=args.device.upper())
        final_input_layer = compiled_model.input(0)
        final_input_shape_tuple = tuple(final_input_layer.shape)
        element_type_numpy = final_input_layer.element_type.to_dtype()
        print(f"Model compiled. Final expected input shape: {final_input_shape_tuple}, Data type: {element_type_numpy}")

    except ImportError: # For LoadImages
        print("Exiting due to LoadImages class import failure.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error during OpenVINO model loading/compilation: {e}", file=sys.stderr)
        return

    # Data Loading
    all_processed_images_np = []
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            # Stride for LoadImages, 32 is common for YOLO. ONNX model itself doesn't store stride.
            # img_size for LoadImages should match the ONNX model's expected input size (args.img_size).
            dataset_loader = LoadImages(args.dataset_path, img_size=args.img_size, stride=32, auto=False)
            for path, img_processed, img0, vid_cap, s_shape in dataset_loader:
                # img_processed from LoadImages is CHW, RGB, uint8 (this is the expected format from utils.datasets.LoadImages)
                img_normalized_np = img_processed.astype(element_type_numpy) / 255.0
                all_processed_images_np.append(img_normalized_np)

            if not all_processed_images_np:
                print(f"Error: No images found/loaded from: {args.dataset_path}", file=sys.stderr)
                return
            print(f"Loaded {len(all_processed_images_np)} images, processed to CHW, RGB, normalized numpy arrays.")
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return
    else:
        print(f"Using dummy data for benchmarking. Shape: {final_input_shape_tuple}, Type: {element_type_numpy}")
        dummy_input_batch_np = numpy.random.rand(*final_input_shape_tuple).astype(element_type_numpy)

    # Batch provider for real data
    current_image_idx = 0
    def get_real_batch_from_preloaded_np():
        nonlocal current_image_idx
        # Check if final_input_shape_tuple is available
        if final_input_shape_tuple is None:
            raise ValueError("Model's final input shape not determined. Cannot create batch.")

        batch_frames_np = numpy.zeros(final_input_shape_tuple, dtype=element_type_numpy)

        for i in range(args.batch_size):
            img_np = all_processed_images_np[current_image_idx % len(all_processed_images_np)]
            # Ensure the single image matches the spatial and channel dims of the batch shape
            if img_np.shape != final_input_shape_tuple[1:]:
                 # Attempt to resize if spatial dimensions don't match (simple resize, might need letterboxing for accuracy)
                 print(f"Warning: Image shape {img_np.shape} does not match model's expected spatial/channel dims {final_input_shape_tuple[1:]}. Attempting resize.", file=sys.stderr)
                 img_resized = cv2.resize(img_np.transpose(1,2,0), (final_input_shape_tuple[3], final_input_shape_tuple[2])).transpose(2,0,1)
                 batch_frames_np[i] = img_resized
            else:
                batch_frames_np[i] = img_np
            current_image_idx += 1
        return batch_frames_np

    # Warm-up Phase
    try:
        print(f"\nStarting {args.warmup_runs} warm-up runs (ONNX model with OpenVINO Runtime)...")
        for _ in range(args.warmup_runs):
            if args.dataset_path:
                batch_data_np = get_real_batch_from_preloaded_np()
                compiled_model.infer_new_request({final_input_layer.any_name: batch_data_np})
            else:
                compiled_model.infer_new_request({final_input_layer.any_name: dummy_input_batch_np})
        print("Warm-up runs completed.")
    except Exception as e:
        print(f"Error during warm-up runs: {e}", file=sys.stderr)
        return

    # Benchmark Phase
    try:
        print(f"\nStarting {args.benchmark_runs} benchmark runs...")
        timings = []
        for i in range(args.benchmark_runs):
            start_time = time.perf_counter()
            if args.dataset_path:
                batch_data_np = get_real_batch_from_preloaded_np()
                compiled_model.infer_new_request({final_input_layer.any_name: batch_data_np})
            else:
                compiled_model.infer_new_request({final_input_layer.any_name: dummy_input_batch_np})
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{args.benchmark_runs} benchmark runs...")
        print("Benchmark runs completed.")
    except Exception as e:
        print(f"Error during benchmark runs: {e}", file=sys.stderr)
        return

    # Results Calculation & Output
    if not timings:
        print("No benchmark timings recorded.", file=sys.stderr)
        return

    avg_time_s = sum(timings) / len(timings)
    avg_time_ms = avg_time_s * 1000
    fps = 1 / avg_time_s if avg_time_s > 0 else 0

    print("\n--- ONNX + OpenVINO Runtime Benchmark Results ---")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path if args.dataset_path else 'Dummy Data'}")
    print(f"Input Shape (used for benchmark, from compiled model): {final_input_shape_tuple}")
    print(f"Device: {args.device.upper()}")
    print(f"Number of warm-up runs: {args.warmup_runs}")
    print(f"Number of benchmark runs: {args.benchmark_runs}")
    print("-------------------------------------------------")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("-------------------------------------------------\n")

if __name__ == '__main__':
    main()
