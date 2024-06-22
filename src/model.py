"""Model module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MultibinRegressor(nn.Module):
    """Multibin regressor model."""

    def __init__(
        self,
        backbone: Literal["resnet18", "mobilenetv3-small"] = "mobilenetv3",
        n_bins: int = 2,
    ) -> None:
        """Initialize the model."""
        assert backbone in [
            "resnet18",
            "mobilenetv3-small",
        ], f"Backbone {backbone} not supported."
        assert n_bins > 1, "Number of bins must be greater than 1."
        super().__init__()

        if backbone == "resnet18":
            from torchvision.models import resnet18

            self.backbone = resnet18(pretrained=True)
        elif backbone == "mobilenetv3-small":
            from torchvision.models import mobilenet_v3_small

            self.backbone = mobilenet_v3_small(pretrained=True)

        self.in_features = self._get_in_features(self.backbone)

        # cut off the last layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # orientation head
        self.ori = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, n_bins * 2),  # 2 for sin and cos
        )

        # orientation confidence head
        self.conf = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, n_bins),  # 1 for confidence
        )

        # dimension head
        self.dim = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 3),  # 3 for length, width, height
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.backbone(x)
        x = x.reshape(-1, self.in_features)

        # orientation
        ori: torch.Tensor = self.ori(x)
        ori = ori.reshape(-1, self.n_bins, 2)
        ori = F.normalize(ori, dim=2)

        # orientation confidence
        conf: torch.Tensor = self.conf(x)

        # dimension
        dim: torch.Tensor = self.dim(x)

        return ori, conf, dim

    def _get_in_features(self, backbone: nn.Module) -> int:
        """Get the number of input features."""
        backbone_name = backbone.__class__.__name__.lower()
        assert backbone_name in [
            "resnet18",
            "mobilenetv3",
        ], f"Backbone {backbone_name} not supported."

        in_features = {
            "resnet": (lambda: backbone.fc.in_features * 7 * 7),  # 512 * 7 * 7 = 25088
            "mobilenetv3-small": (
                lambda: (backbone.classifier[0].in_features) * 7 * 7
            ),  # 576 * 7 * 7 = 28416
        }

        return in_features[backbone_name]()
