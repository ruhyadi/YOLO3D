# YOLO3D—Monocular 3D Object Detection with YOLO

## Introduction

YOLO3D is a dual-stage detector-regressor model for monocular 3D object detection. The first stage is a 2D object detector that detects 2D bounding boxes of objects in the image. The second stage is a 3D object regressor that regresses the 2D bounding boxes to 3D bounding boxes. The 3D bounding boxes are represented as 3D boxes with 3D **location (x, y, z)**, **dimension (tx, ty, tz)**, **and orientation (theta)**.

In this repository, we provide the implementation of YOLO3D with the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) as the 2D object detector and MultiBin by [Mousavian *et al*](https://arxiv.org/abs/1612.00496). as the 3D object regressor. We implement the model in PyTorch and provide the training and evaluation scripts for the [KITTI](https://www.cvlibs.net/datasets/kitti/) dataset. We also provide the ready-to-use pre-trained weights that serve with [FastAPI](https://fastapi.tiangolo.com/) for inference.
