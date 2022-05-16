"""
KITTI Regressor Model
"""
import torch
from torch import nn
from pytorch_lightning import LightningModule

from src.models.components.kitti_net import (RegressorNet, 
                                            OrientationLoss,
                                            get_in_features)

class KITTIModule(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        lr: float = 0.001,
        momentum: float = 0.9,
        bins: int = 2,
        w: float = 0.4,
        alpha: float = 0.6,
        ):
        super().__init__()

        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # init model
        self.net = RegressorNet(self.hparams.backbone, self.hparams.bins)

        # loss functions
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = OrientationLoss

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, 
            momentum=self.hparams.momentum
        )

    def step(self, batch):
        x, y = batch
        
        # convert to float
        x = x.float()
        truth_orient = y['Orientation'].float()
        truth_conf = y['Confidence'].float()
        truth_dim = y['Dimensions'].float()

        # predict y_hat
        preds = self(x)
        [orient, conf, dim] = preds

        # compute loss
        orient_loss = self.orient_loss_func(orient, truth_orient, truth_conf)
        dim_loss = self.dim_loss_func(dim, truth_dim)
        truth_conf = torch.max(truth_conf, dim=1)[1]
        conf_loss = self.conf_loss_func(conf, truth_conf)
        loss_theta = conf_loss + self.hparams.w * orient_loss
        loss = self.hparams.alpha * dim_loss + loss_theta

        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}



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