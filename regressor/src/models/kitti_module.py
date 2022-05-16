"""
KITTI Regressor Model
"""
from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.kitti_net import (RegressorNet, 
                                            OrientationLoss,
                                            get_in_features)

class KITTIModule(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        lr: float = 0.001,
        bins: int = 2,
        w: float = 0.4,
        alpha: float = 0.6,
        ):
        super().__init__()

        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # init model
        self.model = RegressorNet(self.hparams.backbone, self.hparams.bins)

        # loss functions
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = OrientationLoss



if __name__ == '__main__':
    
    from torchvision import models

    ResNet = models.resnet18(pretrained=True)
    VGGNet = models.vgg11(pretrained=True)

    # resnet_in_features = ResNet.fc.in_features
    # vgg_in_features = VGGNet.classifier[0].in_features

    # print(f'ResNet in features: {resnet_in_features}')
    # print(f'VGG in features: {vgg_in_features}')
    # print(ResNet)

    in_features = get_in_features(VGGNet)
    print(in_features)