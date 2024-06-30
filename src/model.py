"""Model module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer, lr_scheduler


class LitMultibinsRegressor(LightningModule):
    """Lightning module for multibins regressor model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler = None,
        omage: float = 0.4,
        alpha: float = 0.6,
        compile: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters(logger=False, ignore="model")
        self.model = model

        # loss fn
        self.ori_lf = ori_loss_fn
        self.conf_lf = nn.CrossEntropyLoss()
        self.dims_lf = nn.MSELoss()

    def setup(self, stage: str) -> None:
        """Setup the model."""
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.model(x)

    def step(self, batch: torch.Tensor) -> List[Dict, Tuple, Tuple]:
        """Forward pass through the model and compute the loss."""
        x, *targets = batch

        # targets
        ori_target, conf_target, dims_target = targets

        # forward pass
        preds = self(x)
        ori_pred, conf_pred, dims_pred = preds

        # compute loss
        ori_loss = self.ori_lf(ori_pred, ori_target, conf_target)
        dims_loss = self.dims_lf(dims_pred, dims_target)
        conf_target = torch.max(conf_target, dim=1)[1]
        conf_loss = self.conf_lf(conf_pred, conf_target)

        # compute theta loss -> see paper
        theta_loss = (self.hparams.omage * ori_loss) + conf_loss
        loss = (self.hparams.alpha * dims_loss) + theta_loss

        return [
            {
                "loss": loss,
                "theta_loss": theta_loss,
                "ori_loss": ori_loss,
                "conf_loss": conf_loss,
                "dims_loss": dims_loss,
            },
            preds,
            targets,
        ]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Training step."""
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "train/loss": loss["loss"],
                "train/theta_loss": loss["theta_loss"],
                "train/ori_loss": loss["ori_loss"],
                "train/conf_loss": loss["conf_loss"],
                "train/dims_loss": loss["dims_loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss["loss"], "preds": preds, "targets": targets}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Validation step."""
        loss, preds, targets = self.step(batch)

        # logging
        self.log_dict(
            {
                "val/loss": loss["loss"],
                "val/theta_loss": loss["theta_loss"],
                "val/ori_loss": loss["ori_loss"],
                "val/conf_loss": loss["conf_loss"],
                "val/dim_loss": loss["dim_loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss["loss"], "preds": preds, "targets": targets}

    def configure_optimizers(self) -> Optimizer:
        """Configure training optimizer and scheduler."""
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )  # type: Optimizer
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 5,
                },
            }

        return {"optimizer": optimizer}


class MultibinsRegressor(nn.Module):
    """Multibins regressor model."""

    def __init__(
        self,
        backbone: Literal["resnet18", "mobilenetv3-small"] = "mobilenetv3-small",
        n_bins: int = 2,
    ) -> None:
        """Initialize the model."""
        assert backbone in [
            "resnet18",
            "mobilenetv3-small",
        ], f"Backbone {backbone} not supported."
        assert n_bins > 1, "Number of bins must be greater than 1."
        super().__init__()
        self.n_bins = n_bins

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
            nn.Linear(256, self.n_bins * 2),  # 2 for sin and cos
        )

        # orientation confidence head
        self.conf = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, self.n_bins),  # 1 for confidence
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
            "resnet",  # resnet18
            "mobilenetv3",
        ], f"Backbone {backbone_name} not supported."

        in_features = {
            "resnet": (lambda: backbone.fc.in_features * 7 * 7),  # 512 * 7 * 7 = 25088
            "mobilenetv3": (
                lambda: (backbone.classifier[0].in_features) * 7 * 7
            ),  # 576 * 7 * 7 = 28416
        }

        return in_features[backbone_name]()


def ori_loss_fn(
    ori_pred: torch.Tensor, ori_target: torch.Tensor, conf_target: torch.Tensor
) -> torch.Tensor:
    """
    Orientation loss function. Inspired from Mousavian et al. (2017).
    """
    batch_size = ori_pred.size()[0]
    indexes = torch.max(conf_target, dim=1)[1]

    # get the orientation prediction for the max confidence bin
    ori_pred = ori_pred[torch.arange(batch_size), indexes]
    ori_target = ori_target[torch.arange(batch_size), indexes]

    # compute the orientation loss
    theta_diff_target = torch.atan2(ori_target[:, 1], ori_target[:, 0])
    theta_diff_pred = torch.atan2(ori_pred[:, 1], ori_pred[:, 0])
    theta_diff = -1 * torch.cos(theta_diff_target - theta_diff_pred).mean()

    return theta_diff
