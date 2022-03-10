"""
Script for regressor model generator with pytorch lightning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, model_select='resnet18', bins=2, w=0.4, lr=0.0001, alpha=0.6):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.bins = bins
        self.w = w
        self.learning_rate = lr
        self.alpha = alpha

        # loss function
        self.conf_loss_func = nn.CrossEntropyLoss()
        self.dim_loss_func = nn.MSELoss()
        self.orient_loss_func = OrientationLoss

        # base model
        self.model = model_factory(model_select)[0]
        self.in_features = model_factory(model_select)[1]

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.in_features)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

    def training_step(self, batch, batch_idx):
        # x = data, y = label
        x, labels = batch
        
        # convert x batch to float
        x = x.float()

        # extracting ground-truth from labels, and convert to float
        truth_orient = labels['Orientation'].float()
        truth_conf = labels['Confidence'].float()
        truth_dim = labels['Dimensions'].float()

        # predict y_hat
        # TODO: maybe using forward()
        [orient, conf, dim] = self(x)

        # commpute loss
        orient_loss = self.orient_loss_func(orient, truth_orient, truth_conf)
        dim_loss = self.dim_loss_func(dim, truth_dim)
        
        truth_conf = torch.max(truth_conf, dim=1)[1]
        conf_loss = self.conf_loss_func(conf, truth_conf)

        loss_theta = conf_loss + self.w*orient_loss
        loss = self.alpha*dim_loss + loss_theta

        # log to tensorboard
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        In validation_step we use batch and batch_idx from validation data
        """
        results = self.training_step(batch, batch_idx)
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        
        # log to tensorboard
        self.log('val_loss', avg_val_loss)
        return {'val_loss': avg_val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        
        return optimizer

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    """
    Orientation loss function
    """
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

def model_factory(model_select):

    # resnet light
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(512, 512)

    # resnet18
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))

    # vgg11
    vgg11 = models.vgg11(pretrained=True)
    vgg11 = vgg11.features

    # store [model, in_features]
    model = {
        'resnet': [resnet, 512],
        'resnet18': [resnet18, 512 * 7 * 7],
        'vgg11': [vgg11, 512 * 7 * 7]
    }

    return model[model_select]

if __name__ == '__main__':
    print('test')
