# YOLO3D: Monocular 3D Object Detection based on YOLO
YOLO3D is dual-stage monocular 3D object detection based on YOLO and [MultiBin](https://arxiv.org/abs/1612.00496v2) architecture. The first stage is 2D object detection done with YOLO model. The second stage is 3D bounding box regression done with MultiBin model. YOLO3D built on top of [PyTorch](https://pytorch.org/), [ONNX](https://onnxruntime.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt). The YOLO model and MultiBin model are trained and tested on [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset. This repository contains code for training (MultiBin), testing and inference the model.

## 💌 Acknowledgement
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX): YOLOX is a high-performance anchor-free YOLO. We use YOLOX as the 2D object detection model.
- [lzccccc/3d-bounding-box-estimation-for-autonomous-driving](https://github.com/lzccccc/3d-bounding-box-estimation-for-autonomous-driving): Thanks for the plotting function in `src/utils/plotting.py`.
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template): Hydra and PyTorch Lightning are awesome, thanks for the template.
- [Mousavian et al. (2017)](https://arxiv.org/abs/1612.00496): Thanks for the MultiBin paper, we use it as the regression model in YOLO3D.