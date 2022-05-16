"""
KITTI Regressor Model
"""
import torch
from torch import nn

class RegressorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bins: int,
    ):
        super().__init__()

        # init model
        self.backbone = backbone
        self.model, self.in_features = get_model(self.backbone)
        self.bins = bins

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.bins),
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

def get_model(backbone: nn.Module):
    """
    Get truncated model and in_features
    """

    # list of support model name
    # TODO: add more models 
    list_model = ['resnet', 'vgg']
    model_name = str(backbone.__class__.__name__).lower()
    assert model_name in list_model, f"Model not support, please choose {list_model}"

    # TODO: change if else with attributes
    in_features = None
    model = None
    if model_name == 'resnet':
        in_features = backbone.fc.in_features
        model = nn.Sequential(*(list(backbone.children())[:-2]))
    elif model_name == 'vgg':
        in_features = backbone.classifier[0].in_features
        model = backbone.features

    return [model, in_features]


if __name__ == '__main__':
    
    from torchvision.models import resnet18

    backbone = resnet18(pretrained=False)
    model = RegressorNet(backbone, 2)

    print(model)