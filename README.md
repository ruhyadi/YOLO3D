# YOLO3D
3D Object Detection with YOLOv5 for Autonomous Driving Vehicle

## Installation
1. Create Conda Env
```
conda create -n yolo3d python=3.8 numpy
```
2. Install pyTorch and torchvision
Download `.whl` from [Nelson Liu](https://cs.stanford.edu/~nfliu/files/pytorch/whl/torch_stable.html) for unsupported GPU (old GPU). Install with pip:
```
pip install ./torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
pip install ./torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl
```
3. Install Requirements
```
pip install -r requirements.txt
```

## Inference
### Inference with `detect.py`
```
python detect.py \
    --weights yolov5s.pt \
    --source data/images \
    --classes 0 1 2 \
    --project runs/detect/
```

## Reference
- [YOLOv5 by ultralytics](https://github.com/ultralytics/yolov5)