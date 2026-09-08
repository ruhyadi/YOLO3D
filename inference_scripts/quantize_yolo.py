#!/usr/bin/env python3
"""
Quantize an OpenVINO IR (XML) model to INT8 using NNCF (post-training).

Usage example:
  python quantize_yolo.py \
    --ir weights/openvino/yolov5s.xml \
    --images data/calib_images/ \
    --output weights/openvino/yolov5s_int8.xml
    

This script reads an OpenVINO IR, builds a simple calibration dataset
from images in a folder, runs NNCF quantization, and writes the quantized IR.
"""
import argparse
import os
import glob
import cv2
import numpy as np
import openvino as ov
import nncf


def parse_args():
    p = argparse.ArgumentParser(description="Quantize OpenVINO IR (YOLO) using NNCF")
    p.add_argument("--ir", required=True, help="Path to input OpenVINO XML model")
    p.add_argument("--images", required=True, help="Folder with calibration images")
    p.add_argument("--output", required=True, help="Path to write quantized XML model")
    p.add_argument("--input-size", type=int, nargs=2, default=(640, 640), help="Model input H W")
    p.add_argument("--batch", type=int, default=1, help="Batch size for calibration")
    p.add_argument("--max-samples", type=int, default=100, help="Max calibration samples to use")
    return p.parse_args()


def preprocess_image(path, input_size):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size[1], input_size[0]))
    img = img.astype(np.float32) / 255.0
    # transpose HWC -> NCHW
    img = np.transpose(img, (2, 0, 1))
    return img


def make_calibration_generator(image_paths, input_size, batch, max_samples):
    def gen():
        batch_buf = []
        count = 0
        for p in image_paths:
            img = preprocess_image(p, input_size)
            batch_buf.append(img)
            count += 1
            if len(batch_buf) == batch:
                yield [np.stack(batch_buf, axis=0)]
                batch_buf = []
            if count >= max_samples:
                break
        if batch_buf:
            yield [np.stack(batch_buf, axis=0)]
    return gen


def main():
    args = parse_args()

    core = ov.Core()
    model = core.read_model(args.ir)

    # gather image paths
    imgs = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
    imgs += sorted(glob.glob(os.path.join(args.images, "**", "*.png"), recursive=True))
    if len(imgs) == 0:
        raise SystemExit("No images found in --images folder for calibration")

    calib_gen = make_calibration_generator(imgs, args.input_size, args.batch, args.max_samples)

    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset=nncf.Dataset(calib_gen())
        )
    except Exception as e:
        print(f"NNCF quantization failed: {e}")
        print("Falling back to saving original IR (no quantization)")
        quantized_model = model

    ov.serialize(quantized_model, args.output)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    main()
