# YOLO For 3D Object Detection


## Training 
```
python train_vgg.py \
    --dataset_path /home/didi/Repository/RTYOLO3D/dataset/KITTI/training/ \
    --weights /home/didi/Repository/RTYOLO3D/weights \
    --epochs 100 \
    --batch_size 50 \
    --num_workers 4 \
```

```
python train_vgg.py \
    --dataset_path /home/didi/Repository/RTYOLO3D/dataset/KITTI/training/ \
    --weights /home/didi/Repository/RTYOLO3D/weights \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 4 \
```