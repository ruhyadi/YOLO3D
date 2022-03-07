"""
Script for training Regressor Model
"""
import argparse
import os
import sys
from pathlib import Path
import glob

from script.Dataset import Dataset

import torch
import torchvision
from torch.utils import data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def train(
    epochs=10,
    batch_size=32,
    alpha=0.6,
    w=0.4,
    num_workers=4,
    train_path=ROOT / 'dataset/KITTI/training'
    ):

    # directory
    dataset = Dataset(train_path)

    params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }

    # generator = data.DataLoader(dataset, **params)

if __name__ == '__main__':
    train_path=str(ROOT / 'dataset/KITTI/training')
    dataset = Dataset(train_path)
    
    for i in range(len(dataset)):
        img = dataset[i][0]
        label = dataset[i][1]

        print(label)

        if i > 3:
            break




