"""
Multibin base model inherit from pytorch.
"""

import rootutils

ROOT = rootutils.autosetup()

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MultibinBase(nn.Module):
    """Multibin regressor module inspired from Mousavian et al. (2017)."""

    def __init__(self, backbone: nn.Module, n_bins: int = 2) -> None:
        """Initialize multibin regressor."""
        super().__init__()

        self.in_features = self.get_in_features(backbone)
        self.model = nn.Sequential(*(list(backbone.children())[:-2]))
        self.n_bins = n_bins

        # orientation head for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.n_bins * 2),  # 4 bins
        )

        # confident head for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, self.n_bins),
        )

        # dimension head for dimension estimation
        self.dimensions = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3),  # x, y, z
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the multibin regressor model."""
        # forward pass of backbone
        x = self.model(x)
        x = x.reshape(-1, self.in_features)

        # forward pass of orientation head
        ori = self.orientation(x)
        ori = ori.reshape(-1, self.n_bins, 2)
        ori = F.normalize(ori, dim=2)

        # forward pass of confidence head
        conf = self.confidence(x)

        # forward pass of dimensions head
        dims = self.dimensions(x)

        return ori, conf, dims

    def get_in_features(self, backbone: nn.Module) -> int:
        """Get the number of features of the backbone."""
        backbone_name = backbone.__class__.__name__.lower()
        assert backbone_name in [
            "resnet",
            "vgg",
            "mobilenetv3",
        ], f"Backbone {backbone_name} not supported."

        in_features = {
            "resnet": (lambda: backbone.fc.in_features * 7 * 7),  # 512 * 7 * 7 = 25088
            "vgg": (lambda: backbone.classifier[0].in_features),  # 512 * 7 * 7 = 25088
            # 'mobilenetv3_large': (lambda: (net.classifier[0].in_features) * 7 * 7), # 960 * 7 * 7 = 47040
            "mobilenetv3": (
                lambda: (backbone.classifier[0].in_features) * 7 * 7
            ),  # 576 * 7 * 7 = 28416
        }

        return in_features[backbone_name]()
