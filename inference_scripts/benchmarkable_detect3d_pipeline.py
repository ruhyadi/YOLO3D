# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import numpy as np
import cv2 # For image operations if any are done outside LoadImages
from pathlib import Path
import sys

# Add project root for custom module imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # Assuming this script is in inference_scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# OpenVINO
try:
    import openvino as ov
except ImportError:
    print("Warning: OpenVINO runtime not found. OpenVINO model types will not be available.")
    ov = None # Define ov as None if import fails

# Project-specific modules (attempt imports, handle if not found initially)
try:
    from models.yolo import Model as YOLOModel # For PyTorch YOLO
    from models.common import DetectMultiBackend # For loading various YOLO backends
    from script.Model import ResNet18 # For PyTorch Regressor
    from torchvision.models import resnet18 as torchvision_resnet18 # Base for custom ResNet18
    from utils.datasets import LoadImages # Will be used by the calling script
    from utils.general import non_max_suppression, scale_coords, check_img_size
    from utils.augmentations import letterbox # Corrected import for letterbox
    from utils.torch_utils import select_device
    from script.Dataset import DetectedObject, generate_bins # For 3D processing
    from script.ClassAverages import ClassAverages
except ImportError as e:
    print(f"Warning: Initial import failed for some project modules: {e}. Ensure PYTHONPATH is set or script is in correct location.")
    YOLOModel, DetectMultiBackend, ResNet18, torchvision_resnet18, LoadImages = None,None,None,None,None
    non_max_suppression, scale_coords, check_img_size, select_device = None,None,None,None # letterbox removed here
    letterbox = None # Explicitly set letterbox to None here as well
    DetectedObject, generate_bins, ClassAverages = None,None,None


def load_yolo_detector(weights_path, cfg_path, device_str, model_type='pytorch_native', img_size=(640,640), batch_size=1):
    print(f"Loading YOLO detector ({model_type}) from: {weights_path} with cfg: {cfg_path} on device: {device_str}")
    if YOLOModel is None and model_type in ['pytorch_native', 'pytorch_compile_openvino']:
        raise ImportError("YOLOModel class from models.yolo is not available for PyTorch types.")
    if ov is None and model_type in ['onnx_openvino', 'ir_openvino']:
        raise ImportError("OpenVINO runtime (ov) not available for OpenVINO types.")

    try:
        if model_type == 'pytorch_native':
            device = torch.device(device_str)
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            final_model = None
            # Extract names from checkpoint before rebuilding model from cfg
            names_from_checkpoint = None
            if hasattr(checkpoint, 'names'):
                names_from_checkpoint = checkpoint.names
            elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'names'):
                names_from_checkpoint = checkpoint['model'].names
            if hasattr(checkpoint, 'yaml') and hasattr(checkpoint, 'model') and hasattr(checkpoint, 'eval'):
                final_model = checkpoint.to(device).eval()
                if cfg_path:
                    temp_model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).to(device)
                    temp_model_from_cfg.load_state_dict(final_model.state_dict())
                    final_model = temp_model_from_cfg.eval()
            elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                loaded_sub_model = checkpoint['model']
                if hasattr(loaded_sub_model, 'yaml_file') or hasattr(loaded_sub_model, 'yaml'):
                    final_model = loaded_sub_model.to(device).float().eval()
                    if cfg_path:
                        temp_model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).to(device)
                        temp_model_from_cfg.load_state_dict(final_model.state_dict())
                        final_model = temp_model_from_cfg.eval()
                else:
                    model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).to(device)
                    model_from_cfg.load_state_dict(loaded_sub_model.state_dict())
                    final_model = model_from_cfg.eval()
            elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
                state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
                state_dict = checkpoint[state_dict_key]
                model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).to(device)
                model_from_cfg.load_state_dict(state_dict)
                final_model = model_from_cfg.eval()
            else:
                model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).to(device)
                model_from_cfg.load_state_dict(checkpoint)
                final_model = model_from_cfg.eval()
            if final_model is None: raise ValueError("Could not load PyTorch native model.")
            # Restore class names from the original checkpoint (lost when rebuilding from cfg)
            if names_from_checkpoint is not None:
                final_model.names = names_from_checkpoint
            print(f"PyTorch native YOLO model loaded successfully on {device_str}.")
            return final_model

        elif model_type == 'pytorch_compile_openvino':
            device_cpu = torch.device('cpu')
            checkpoint = torch.load(weights_path, map_location=device_cpu, weights_only=False)
            model_on_cpu = None
            # Extract names from checkpoint before rebuilding model from cfg
            names_from_checkpoint = None
            if hasattr(checkpoint, 'names'):
                names_from_checkpoint = checkpoint.names
            elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'names'):
                names_from_checkpoint = checkpoint['model'].names
            if hasattr(checkpoint, 'yaml') and hasattr(checkpoint, 'model') and hasattr(checkpoint, 'eval'):
                model_on_cpu = checkpoint.cpu().eval()
                if cfg_path:
                    temp_model_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
                    temp_model_cfg.load_state_dict(model_on_cpu.state_dict())
                    model_on_cpu = temp_model_cfg.eval()
            elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'): # More complete handling
                loaded_sub_model = checkpoint['model'].cpu()
                if hasattr(loaded_sub_model, 'yaml_file') or hasattr(loaded_sub_model, 'yaml'):
                    model_on_cpu = loaded_sub_model.float().eval()
                    if cfg_path:
                        temp_model_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
                        temp_model_cfg.load_state_dict(model_on_cpu.state_dict())
                        model_on_cpu = temp_model_cfg.eval()
                else:
                    model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
                    model_from_cfg.load_state_dict(loaded_sub_model.state_dict())
                    model_on_cpu = model_from_cfg.eval()
            elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
                state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
                state_dict = checkpoint[state_dict_key]
                model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
                model_from_cfg.load_state_dict(state_dict)
                model_on_cpu = model_from_cfg.eval()
            else:
                model_from_cfg = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
                model_from_cfg.load_state_dict(checkpoint)
                model_on_cpu = model_from_cfg.eval()

            if model_on_cpu is None: raise ValueError("Could not load PyTorch model on CPU for compile.")
            # Restore class names from the original checkpoint (lost when rebuilding from cfg)
            if names_from_checkpoint is not None:
                model_on_cpu.names = names_from_checkpoint
            print(f"PyTorch YOLO model loaded on CPU, compiling with OpenVINO backend for {device_str.upper()}...")
            compiled_model = torch.compile(model_on_cpu, backend="openvino", options={"device": device_str.upper()})
            print("PyTorch YOLO model compiled with OpenVINO backend successfully.")
            return compiled_model

        elif model_type in ['onnx_openvino', 'ir_openvino']:
            core = ov.Core()
            print(f"Reading OpenVINO compatible model ({('ONNX' if model_type == 'onnx_openvino' else 'IR')}) from: {weights_path}")
            ov_model = core.read_model(weights_path)
            try:
                input_name = ov_model.input(0).any_name
                # Use fixed batch size
                new_shape = ov.PartialShape([1, 3, img_size[0], img_size[1]])
                ov_model.reshape({input_name: new_shape})
                print(f"Reshaped OpenVINO YOLO model input to {new_shape} (fixed batch size: 1)")
            except Exception as e:
                print(f"Warning: Could not reshape OpenVINO model. Using default shape. Error: {e}", file=sys.stderr)
            print(f"Compiling OpenVINO model for device: {device_str.upper()}")
            compiled_model = core.compile_model(ov_model, device_name=device_str.upper())
            print("OpenVINO model compiled successfully.")
            return compiled_model
        else:
            raise ValueError(f"Unsupported yolo_model_type: {model_type}")
    except Exception as e:
        print(f"Error loading/compiling YOLO detector ({model_type}): {e}", file=sys.stderr)
        return None

def load_regressor_model(weights_path, model_name_str, device_str, model_type='pytorch_native', img_size=(224,224), batch_size=1):
    print(f"Loading Regressor ({model_type}, {model_name_str}) from: {weights_path} on device: {device_str}")
    if model_name_str != 'resnet18':
        print(f"Warning: This function is primarily set up for 'resnet18' based custom regressor. Found: {model_name_str}", file=sys.stderr)

    if ResNet18 is None and model_type in ['pytorch_native', 'pytorch_compile_openvino']:
        raise ImportError("Custom ResNet18 class not available for PyTorch types.")
    if torchvision_resnet18 is None and model_type in ['pytorch_native', 'pytorch_compile_openvino']:
        raise ImportError("Base torchvision.models.resnet18 not available for PyTorch types.")
    if ov is None and model_type in ['onnx_openvino', 'ir_openvino']:
        raise ImportError("OpenVINO runtime (ov) not available for OpenVINO types.")

    try:
        if model_type == 'pytorch_native':
            device = torch.device(device_str)
            base_tv_model = torchvision_resnet18(weights=None, progress=False)
            model = ResNet18(model=base_tv_model).to(device)
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            # ... (robust pkl loading logic as before)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict): model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, torch.nn.Module) and isinstance(checkpoint, ResNet18): model = checkpoint.to(device)
            else: raise ValueError("PKL content not a recognized state_dict or ResNet18 instance.")
            model.eval()
            print(f"PyTorch native Regressor model loaded successfully on {device_str}.")
            return model

        elif model_type == 'pytorch_compile_openvino':
            device_cpu = torch.device('cpu')
            base_tv_model = torchvision_resnet18(weights=None, progress=False)
            model_cpu = ResNet18(model=base_tv_model).to(device_cpu)
            checkpoint = torch.load(weights_path, map_location=device_cpu, weights_only=False)
            # ... (robust pkl loading logic for CPU as before)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: model_cpu.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: model_cpu.load_state_dict(checkpoint['state_dict'])
            else: model_cpu.load_state_dict(checkpoint) # Assume raw state_dict
            model_cpu.eval()
            print(f"PyTorch Regressor model loaded on CPU, compiling with OpenVINO backend for {device_str.upper()}...")
            compiled_model = torch.compile(model_cpu, backend="openvino", options={"device": device_str.upper()})
            print("PyTorch Regressor compiled with OpenVINO backend successfully.")
            return compiled_model

        elif model_type in ['onnx_openvino', 'ir_openvino']:
            core = ov.Core()
            print(f"Reading OpenVINO compatible Regressor model ({('ONNX' if model_type == 'onnx_openvino' else 'IR')}) from: {weights_path}")
            ov_model = core.read_model(weights_path)
            try:
                input_name = ov_model.input(0).any_name
                # Use fixed batch size
                new_shape = ov.PartialShape([batch_size, 3, img_size[0], img_size[1]])
                ov_model.reshape({input_name: new_shape})
                print(f"Reshaped OpenVINO Regressor model input to {new_shape} (fixed batch size: {batch_size})")
            except Exception as e:
                print(f"Warning: Could not reshape OpenVINO Regressor model. Using default shape. Error: {e}", file=sys.stderr)
            print(f"Compiling OpenVINO Regressor model for device: {device_str.upper()}")
            
            # Apply performance optimization config BEFORE compilation
            config = {
                'PERFORMANCE_HINT': 'LATENCY',  # Optimize for low latency (not throughput)
            }
            compiled_model = core.compile_model(ov_model, device_name=device_str.upper(), config=config)
            print("OpenVINO Regressor model compiled successfully.")
            print(f"Applied performance optimization config: {config}")
            
            return compiled_model
        else:
            raise ValueError(f"Unsupported regressor_model_type: {model_type}")
    except Exception as e:
        print(f"Error loading/compiling Regressor model ({model_type}): {e}", file=sys.stderr)
        return None

# COCO to KITTI class name mapping (for OpenVINO models that can't extract names)
COCO_TO_KITTI_MAPPING = {
    0: 'pedestrian',     # person
    2: 'car',
    3: 'cyclist',        # bicycle
    5: 'truck',          # bus -> truck (approximate)
    7: 'truck',
}

def benchmarkable_detect3d(
    yolo_model,
    regressor_model,
    yolo_model_type,
    regressor_model_type,
    input_image_np,     # HWC, BGR, uint8
    img_size_yolo,      # tuple (H, W)
    stride_yolo,
    img_size_regressor, # tuple (H, W)
    device_pytorch,
    device_openvino,
    calib_file_path,
    class_averages,
    angle_bins,
    conf_thres_yolo=0.25,
    iou_thres_yolo=0.45,
    yolo_classes_to_detect=None,
    max_detections_yolo=1000,
    regressor_batch_size=64,
    override_class_names=None
):
    timings = {}
    img_original_shape_hw = input_image_np.shape[:2]

    # Use override_class_names if provided, otherwise extract from model
    if override_class_names is not None:
        names = override_class_names
    else:
        names = None
        if yolo_model_type in ['pytorch_native', 'pytorch_compile_openvino']:
            yolo_model_for_names = yolo_model._model if hasattr(yolo_model, '_model') and hasattr(yolo_model._model, 'names') else yolo_model
            if hasattr(yolo_model_for_names, 'names'): names = yolo_model_for_names.names
        if names is None: 
            # For OpenVINO models, create COCO names with KITTI mapping fallback
            coco_names = [f'class_{i}' for i in range(80)]
            names = coco_names

    time_start_yolo_preproc = time.perf_counter()
    img_yolo_letterboxed = letterbox(input_image_np, new_shape=img_size_yolo, stride=stride_yolo, auto=False)[0]
    img_yolo_chw_rgb = img_yolo_letterboxed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_yolo_chw_rgb_contiguous = np.ascontiguousarray(img_yolo_chw_rgb)
    yolo_input_feed = None
    if yolo_model_type in ['pytorch_native', 'pytorch_compile_openvino']:
        yolo_input_feed = torch.from_numpy(img_yolo_chw_rgb_contiguous)
        yolo_input_feed = yolo_input_feed.to(device_pytorch if yolo_model_type == 'pytorch_native' else torch.device('cpu'))
        yolo_input_feed = yolo_input_feed.float() / 255.0
        if yolo_input_feed.ndimension() == 3: yolo_input_feed = yolo_input_feed.unsqueeze(0)
    elif yolo_model_type in ['onnx_openvino', 'ir_openvino']:
        yolo_input_feed = (img_yolo_chw_rgb_contiguous.astype(np.float32) / 255.0)[np.newaxis, :]
    else: raise ValueError(f"Unsupported yolo_model_type for preprocessing: {yolo_model_type}")
    timings['yolo_preprocessing'] = time.perf_counter() - time_start_yolo_preproc

    yolo_preds_raw = None
    time_start_yolo_infer = time.perf_counter()
    if yolo_model_type == 'pytorch_native':
        with torch.no_grad(): yolo_preds_raw_tuple = yolo_model(yolo_input_feed); yolo_preds_raw = yolo_preds_raw_tuple[0] if isinstance(yolo_preds_raw_tuple, tuple) else yolo_preds_raw_tuple
    elif yolo_model_type == 'pytorch_compile_openvino':
        with torch.no_grad(): yolo_preds_raw_tuple = yolo_model(yolo_input_feed); yolo_preds_raw = yolo_preds_raw_tuple[0] if isinstance(yolo_preds_raw_tuple, tuple) else yolo_preds_raw_tuple
    elif yolo_model_type in ['onnx_openvino', 'ir_openvino']:
        input_name = yolo_model.input(0).any_name
        output_tensor_ov = yolo_model.output(0)
        results = yolo_model.infer_new_request({input_name: yolo_input_feed})
        yolo_preds_raw = torch.from_numpy(results[output_tensor_ov]).to(device_pytorch if device_pytorch.type != 'cuda' else 'cpu')
    timings['yolo_inference'] = time.perf_counter() - time_start_yolo_infer

    time_start_yolo_postproc = time.perf_counter()
    detections_after_nms = non_max_suppression(yolo_preds_raw, conf_thres_yolo, iou_thres_yolo, classes=yolo_classes_to_detect, agnostic=False, max_det=max_detections_yolo)
    processed_detections_for_regressor = []
    if detections_after_nms and detections_after_nms[0] is not None:
        single_image_detections = detections_after_nms[0].cpu()
        scaled_coords_dets = scale_coords(yolo_input_feed.shape[2:], single_image_detections[:, :4], img_original_shape_hw).round()
        for i, det in enumerate(single_image_detections):
            processed_detections_for_regressor.append({"xyxy_scaled": scaled_coords_dets[i].numpy(), "conf": det[4].item(), "cls_idx": int(det[5].item())})
    num_2d_detections = len(processed_detections_for_regressor)
    timings['yolo_postprocessing'] = time.perf_counter() - time_start_yolo_postproc

    timings.update({'regressor_crop_preprocess_total': 0, 'regressor_inference_batch_total': 0, 'regressor_output_decode_total': 0, 'num_regressor_runs': 0})

    if num_2d_detections > 0:
        regressor_input_batch_constructor = []
        skipped_due_to_class = 0
        skipped_due_to_exception = 0
        
        for det_info in processed_detections_for_regressor:
            _t_start = time.perf_counter()
            cls_idx = det_info["cls_idx"]
            
            # Get class name: try direct names lookup, fallback to COCO-to-KITTI mapping for OpenVINO
            if names and 0 <= cls_idx < len(names):
                detected_class_name = names[cls_idx]
            else:
                detected_class_name = f'class_{cls_idx}'
            
            # For OpenVINO models using generic class names, try COCO-to-KITTI mapping
            if detected_class_name.startswith('class_') and yolo_model_type in ['onnx_openvino', 'ir_openvino']:
                if cls_idx in COCO_TO_KITTI_MAPPING:
                    detected_class_name = COCO_TO_KITTI_MAPPING[cls_idx]
            
            if class_averages and not class_averages.recognized_class(detected_class_name):
                timings['regressor_crop_preprocess_total'] += time.perf_counter() - _t_start
                skipped_due_to_class += 1
                continue
            # Convert flat coordinates [x1, y1, x2, y2] to nested format [[x1, y1], [x2, y2]]
            xyxy = [int(c) for c in det_info["xyxy_scaled"]]
            box_2d = [(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])]
            try:
                detected_obj = DetectedObject(input_image_np, detected_class_name, box_2d, calib_file_path)
                regressor_patch_tensor = detected_obj.img
                if regressor_model_type in ['pytorch_native', 'pytorch_compile_openvino']: 
                    regressor_input_batch_constructor.append(regressor_patch_tensor)
                elif regressor_model_type in ['onnx_openvino', 'ir_openvino']: 
                    regressor_input_batch_constructor.append(regressor_patch_tensor.cpu().numpy())
                timings['regressor_crop_preprocess_total'] += time.perf_counter() - _t_start
            except Exception as e:
                print(f"Warning: Failed to create DetectedObject for class '{detected_class_name}' with box {box_2d}: {e}", file=sys.stderr)
                timings['regressor_crop_preprocess_total'] += time.perf_counter() - _t_start
                skipped_due_to_exception += 1
                continue

        timings['num_regressor_runs'] = len(regressor_input_batch_constructor)
        
        if regressor_input_batch_constructor:
            for i in range(0, len(regressor_input_batch_constructor), regressor_batch_size):
                current_batch_list = regressor_input_batch_constructor[i : i + regressor_batch_size]
                if not current_batch_list: continue
                
                actual_batch_size = len(current_batch_list)
                reg_batch_feed = None
                
                if regressor_model_type in ['pytorch_native', 'pytorch_compile_openvino']:
                    reg_batch_feed = torch.stack(current_batch_list)
                    if regressor_model_type == 'pytorch_native': reg_batch_feed = reg_batch_feed.to(device_pytorch)
                    
                elif regressor_model_type in ['onnx_openvino', 'ir_openvino']: 
                    reg_batch_feed = np.stack(current_batch_list)  # Shape: (actual_batch_size, 3, 224, 224)
                    
                    # Pad incomplete batches to match model's expected batch size
                    if actual_batch_size < regressor_batch_size:
                        padding_needed = regressor_batch_size - actual_batch_size
                        padding = np.zeros((padding_needed, *reg_batch_feed.shape[1:]), dtype=reg_batch_feed.dtype)
                        reg_batch_feed = np.vstack([reg_batch_feed, padding])

                _t_infer_start = time.perf_counter()
                if regressor_model_type == 'pytorch_native':
                    with torch.no_grad(): _, _, _ = regressor_model(reg_batch_feed)
                elif regressor_model_type == 'pytorch_compile_openvino':
                    with torch.no_grad(): _, _, _ = regressor_model(reg_batch_feed)
                elif regressor_model_type in ['onnx_openvino', 'ir_openvino']:
                    reg_input_name = regressor_model.input(0).any_name
                    _ = regressor_model.infer_new_request({reg_input_name: reg_batch_feed})
                    # Note: we only used actual_batch_size samples, the padding is discarded
                    
                batch_inference_time = time.perf_counter() - _t_infer_start
                timings['regressor_inference_batch_total'] += batch_inference_time

    annotated_image = input_image_np.copy()
    return annotated_image, timings

if __name__ == '__main__':
    print("benchmarkable_detect3d_pipeline.py structure defined.")
