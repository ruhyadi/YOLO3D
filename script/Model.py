"""
Script for regressor model generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ResNet(nn.Module):
    def __init__(self, model=None, bins=2, w=0.4):
        super(ResNet, self).__init__()
        self.bins = bins
        self.w = w
        model.fc = nn.Linear(512, 512)
        self.model = model

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

class ResNet18(nn.Module):
    def __init__(self, model=None, bins=2, w=0.4):
        super(ResNet18, self).__init__()
        self.bins = bins
        self.w = w
        self.model = nn.Sequential(*(list(model.children())[:-2]))

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins) # 2 bins   
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512 * 7 * 7)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

class VGG11(nn.Module):
    def __init__(self, model=None, bins=2, w=0.4):
        super(VGG11, self).__init__()
        self.bins = bins
        self.w = w
        # vgg11 features
        self.model = model.features

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins)  
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512 * 7 * 7)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

if __name__ == '__main__':
    print('test')
