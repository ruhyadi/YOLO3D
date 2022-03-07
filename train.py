"""
Script for training Regressor Model
"""

import argparse
import os
import sys
from pathlib import Path

from script.Dataset import Dataset
from script.Model import ResNet18, VGG11, OrientationLoss

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, vgg11
from torch.utils import data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# model factory to choose model
model_factory = {
    'resnet18': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet18': ResNet18,
    'vgg11': VGG11
}

def train(
    epochs=10,
    batch_size=32,
    alpha=0.6,
    w=0.4,
    num_workers=4,
    lr=0.0001,
    train_path=ROOT / 'dataset/KITTI/training',
    model_path=ROOT / 'weights',
    model_list='resnet18'
    ):

    # directory
    train_path = str(train_path)
    model_path = str(model_path)

    # dataset
    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path)

    params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }

    # data generator
    data_gen = data.DataLoader(dataset, **params)

    # model
    base_model = model_factory[model_list]
    model = regressor_factory[model_list](model=base_model).cuda()
    
    # optimizer
    opt_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # loss function
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load previous weights
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'[INFO] Using previous model {latest_model} at {first_epoch} epochs')
        print('[INFO] Resuming training...')

    total_num_batches = int(len(dataset)/batch_size)

if __name__ == '__main__':
    train_path=str(ROOT / 'dataset/KITTI/training')
    train()




