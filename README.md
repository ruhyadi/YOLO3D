# YOLO3D

## Installation
Create conda environment
```bash
conda create -n capstone python=3.8 numpy
```
Install depedencies from `requirements.txt`
```bash
cd yolo3d
pip install -r requirements.txt
```
### Custom PyTorch
Install custom pytorch because CUDA support
```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Preparing Dataset

### Download KITTI Dataset
Download KITTI dataset from [official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
```bash
cd /yolo3d/data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
```
Unzip dataset to `yolo3d/data/KITTI`.
```
cd /yolo3d/data
mkdir KITTI
unzip data_object_image_2.zip -d ./KITTI
unzip data_object_label_2.zip -d ./KITTI
unzip data_object_calib.zip -d ./KITTI
```

### Download KITTI YOLO Annotations
Download KITTI YOLO Annotations from Kaggle (currently private dataset) with [Kaggle API](https://www.kaggle.com/docs/api).
```bash
cd /yolo3d/data/KITTI/training
kaggle datasets download -d didiruh/capstone-kitti-annotations
```
unzip and remove `.zip` data
```bash
unzip capstone-kitti-annotations.zip -d ./
rm -rf capstone-kitti-annotations.zip
```
Final dataset directory will be like
```bash
.
└── yolo3d
    ├── data
    │   └── KITTI
    │       └── training
    │           ├── calib   # calib
    │           ├── images  # image_2
    │           ├── label_2 # KITTI Format
    │           ├── labels  # YOLO Format
    │           ├── train_yolo.txt
    │           └── val_yolo.txt
    └── other-files
```

### Prepare `train.yaml`
`.yaml` file used for training configuration. Can be found at `data/*.yaml`. Let's create `kitti.yaml` to training KITTI dataset.
```yaml
path: ./data/KITTI/training  # dataset root dir
train: train_yolo.txt  # train images (relative to 'path')
val: val_yolo.txt  # val images (relative to 'path'

# Classes
nc: 4  # number of classes
names: [ 'car', 'pedestrian', 'cyclist', 'truck' ]  # class names
```

## Training Model
Single GPU
```bash
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 3 \
    --data kitti.yaml \
    --weights yolov5s.pt
```
Multiple GPU ([read more](https://github.com/ultralytics/yolov5/issues/475)). In here I'm training with 4xV100 DGX GPU. 
```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    train.py \
    --batch 64 \
    --epochs 2 \
    --data kitti.yaml \
    --weights yolov5s.pt \
    --device 0,1,2,3
```