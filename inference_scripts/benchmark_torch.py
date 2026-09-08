# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import time
import argparse
from pathlib import Path
import sys
import cv2 # Explicitly import cv2
import random # For selecting images

# Works around a torch._dynamo guard-building crash (AssertionError: sources
# must not be empty) seen on this YOLOv5 model with the installed torch build;
# falls back to eager execution instead of hard-crashing.
torch._dynamo.config.suppress_errors = True

# Add project root to sys.path to allow imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from models.yolo import Model # For loading YOLOv5 model structure
    from utils.datasets import LoadImages # For loading real datasets
except ImportError as e:
    print(f"Error: Could not import 'Model' from 'models.yolo' or 'LoadImages' from 'utils.datasets'. {e}", file=sys.stderr)
    # Dummy classes for script structure viability if imports fail
    class Model:
        def __init__(self, cfg, ch=None, nc=None): raise ImportError("Real Model class not available.")
        def to(self, dev): raise ImportError("Real Model class not available.")
        def load_state_dict(self, sd, strict=False): raise ImportError("Real Model class not available.")
        def eval(self): raise ImportError("Real Model class not available.")
        def __call__(self, x): raise ImportError("Real Model class not available.")
    class LoadImages:
        def __init__(self, p, img_size=640, stride=32, auto=True, vid_stride=1): raise ImportError("Real LoadImages not available.")
        def __iter__(self): return self
        def __next__(self): raise StopIteration
        def __len__(self): return 0

def main():
    parser = argparse.ArgumentParser(description="PyTorch + torch.compile OpenVINO Backend Benchmark Script with Real Dataset")
    parser.add_argument('--weights', type=str, required=True, help='Path to PyTorch model weights file (e.g., weights/yolov5s.pt)')
    parser.add_argument('--cfg', type=str, required=True, help='Path to model configuration YAML file (e.g., models/yolov5s.yaml)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset directory or image file. If None, uses dummy data.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for model input (square image assumed)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warm-up inference runs')
    parser.add_argument('--benchmark_runs', type=int, default=50, help='Number of benchmark inference runs')
    parser.add_argument('--device', type=str, default='CPU', help="OpenVINO device for backend (e.g., 'CPU', 'GPU').")

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: Weights file not found at {args.weights}", file=sys.stderr)
        return
    if not Path(args.cfg).exists():
        print(f"Error: Model config file not found at {args.cfg}", file=sys.stderr)
        return
    if args.dataset_path and not Path(args.dataset_path).exists() and not Path(args.dataset_path).is_file():
        if not (Path(args.dataset_path).parent.exists() and '*' in Path(args.dataset_path).name):
             print(f"Error: Dataset path {args.dataset_path} does not exist or is not a file/glob pattern.", file=sys.stderr)
             return

    # Model Loading - always on CPU before torch.compile with OpenVINO backend
    pytorch_device = torch.device('cpu') # Explicitly use CPU for initial model loading
    model_stride = 32 # Default stride
    model = None # Initialize model variable
    YOLOModel = Model # Alias for clarity

    try:
        print(f"Loading PyTorch model from weights: '{args.weights}' onto CPU for torch.compile...")
        checkpoint = torch.load(args.weights, map_location=pytorch_device, weights_only=False)

        model_on_cpu = None
        if hasattr(checkpoint, 'yaml') and hasattr(checkpoint, 'model') and hasattr(checkpoint, 'eval'): # Ultralytics full model
            print("Checkpoint appears to be a full model object (Ultralytics format). Ensuring it's on CPU.")
            model_on_cpu = checkpoint.cpu().eval()
            # For consistency, if args.cfg is provided, re-instantiate and load state_dict
            if args.cfg:
                print(f"Aligning loaded model with specified CFG: {args.cfg} on CPU.")
                temp_model_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).cpu()
                temp_model_cfg.load_state_dict(model_on_cpu.state_dict())
                model_on_cpu = temp_model_cfg.eval() # Already on CPU and eval

        elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
            print("Checkpoint is a dictionary with a 'model' attribute (nn.Module). Ensuring it's on CPU.")
            loaded_sub_model = checkpoint['model'].cpu() # Ensure sub_model is on CPU
            if hasattr(loaded_sub_model, 'yaml_file') or hasattr(loaded_sub_model, 'yaml'): # If it's a YOLOModel instance
                 model_on_cpu = loaded_sub_model.float().eval() # Already on CPU
                 if args.cfg: # Align with args.cfg if provided
                    print(f"Aligning loaded YOLOModel instance with specified CFG: {args.cfg} on CPU.")
                    temp_model_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).cpu()
                    temp_model_cfg.load_state_dict(model_on_cpu.state_dict())
                    model_on_cpu = temp_model_cfg.eval()
            else: # Generic nn.Module, load its state_dict into our cfg model (which is already .cpu())
                 print("Loaded sub-model is a generic nn.Module. Creating model from CFG on CPU and loading state_dict.")
                 model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).cpu()
                 model_from_cfg.load_state_dict(loaded_sub_model.state_dict())
                 model_on_cpu = model_from_cfg.eval()

        elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
            state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
            print(f"Checkpoint is a dictionary with '{state_dict_key}'. Creating model from CFG on CPU.")
            state_dict = checkpoint[state_dict_key]
            model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).cpu()
            model_from_cfg.load_state_dict(state_dict)
            model_on_cpu = model_from_cfg.eval()
        else: # Assume checkpoint is a raw state_dict
            print("Checkpoint is assumed to be a raw state_dict. Creating model from CFG on CPU.")
            model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).cpu()
            model_from_cfg.load_state_dict(checkpoint)
            model_on_cpu = model_from_cfg.eval()

        if model_on_cpu is None:
            raise ValueError("Could not load model from checkpoint onto CPU.")

        model = model_on_cpu # Assign to the script's 'model' variable

        if hasattr(model, 'stride'):
            model_stride = int(model.stride.max() if isinstance(model.stride, torch.Tensor) else model.stride)
        print(f"PyTorch model loaded successfully on CPU. Using stride: {model_stride}")

    except ImportError: # For Model or LoadImages
        print("Exiting due to Model or LoadImages class import failure.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading PyTorch model for torch.compile: {e}", file=sys.stderr)
        return

    # Model Compilation
    try:
        print(f"Compiling model with torch.compile for OpenVINO backend, device: {args.device.upper()}...")
        compiled_model = torch.compile(model, backend="openvino", options={"device": args.device.upper()})
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Error during torch.compile with OpenVINO backend: {e}", file=sys.stderr)
        print("Please ensure 'openvino-pytorch' is installed.", file=sys.stderr)
        return

    # Data Loading
    all_processed_images_cpu = []
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            dataset_loader = LoadImages(args.dataset_path, img_size=args.img_size, stride=model_stride, auto=False)
            for path, img_processed, img0, vid_cap, s_shape in dataset_loader:
                # img_processed from LoadImages is CHW, RGB, uint8
                img_tensor = torch.from_numpy(img_processed).cpu() # Ensure on CPU
                img_tensor = img_tensor.float() / 255.0 # Normalize to [0.0, 1.0]
                all_processed_images_cpu.append(img_tensor)

            if not all_processed_images_cpu:
                print(f"Error: No images found/loaded from: {args.dataset_path}", file=sys.stderr)
                return
            print(f"Loaded {len(all_processed_images_cpu)} images to CPU memory.")
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return
    else:
        print("Using dummy data for benchmarking (on CPU).")
        dummy_input_batch_cpu = torch.randn(args.batch_size, 3, args.img_size, args.img_size).cpu()

    # Batch provider for real data (from CPU preloaded list)
    current_image_idx = 0
    def get_real_batch_from_preloaded_cpu():
        nonlocal current_image_idx
        batch_frames = []
        for _ in range(args.batch_size):
            img_tensor = all_processed_images_cpu[current_image_idx % len(all_processed_images_cpu)]
            batch_frames.append(img_tensor)
            current_image_idx += 1
        return torch.stack(batch_frames) # Already on CPU

    # Warm-up Phase
    try:
        print(f"\nStarting {args.warmup_runs} warm-up runs (PyTorch with torch.compile OpenVINO backend)...")
        for _ in range(args.warmup_runs):
            with torch.no_grad():
                if args.dataset_path:
                    batch_data = get_real_batch_from_preloaded_cpu()
                    _ = compiled_model(batch_data)
                else:
                    _ = compiled_model(dummy_input_batch_cpu)
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
            with torch.no_grad():
                if args.dataset_path:
                    batch_data = get_real_batch_from_preloaded_cpu()
                    _ = compiled_model(batch_data)
                else:
                    _ = compiled_model(dummy_input_batch_cpu)
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

    print("\n--- PyTorch + torch.compile OpenVINO Benchmark Results ---")
    print(f"Model: {args.weights}")
    print(f"Config: {args.cfg}")
    print(f"Dataset: {args.dataset_path if args.dataset_path else 'Dummy Data'}")
    print(f"Image Size (model input): {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"OpenVINO Device: {args.device.upper()}") # This is the OpenVINO backend device
    print(f"Number of warm-up runs: {args.warmup_runs}")
    print(f"Number of benchmark runs: {args.benchmark_runs}")
    print("----------------------------------------------------------")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("----------------------------------------------------------\n")

if __name__ == '__main__':
    main()
