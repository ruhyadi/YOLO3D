"""
Multibin regressor pytorch lightning module.
"""

import rootutils

ROOT = rootutils.autosetup()

from typing import Any, List

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer, lr_scheduler


class MultibinModule(LightningModule):
    """Multibin regressor module inspired from Mousavian et al. (2017)."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler = None,
        omega: float = 0.4,
        alpha: float = 0.6,
        compile: bool = True,
    ) -> None:
        """Initialize multibin regressor lightning module."""
        super().__init__()
        self.save_hyperparameters(logger=False, ignore="net")
        self.net = net

        # loss functions
        self.ori_lf = orientation_loss
        self.conf_lf = nn.CrossEntropyLoss()
        self.dim_lf = nn.MSELoss()

    def setup(self, stage: str) -> None:
        """Setup the model with Lightning hook."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def step(self, batch: torch.Tensor) -> List:
        """Forward pass through the model and compute the loss."""
        x, *targets = batch

        # targets
        ori_target, conf_target, dim_target = targets

        # forward pass
        preds = self(x)
        ori_pred, conf_pred, dim_pred = preds

        # compute loss
        ori_loss = self.ori_lf(ori_pred, ori_target, conf_target)
        dim_loss = self.dim_lf(dim_pred, dim_target)
        conf_target = torch.max(conf_target, dim=1)[1]
        conf_loss = self.conf_lf(conf_pred, conf_target)

        # compute theta loss -> see paper
        theta_loss = (self.hparams.omega * ori_loss) + conf_loss
        loss = (self.hparams.alpha * dim_loss) + theta_loss

        return [
            {
                "loss": loss,
                "theta_loss": theta_loss,
                "ori_loss": ori_loss,
                "conf_loss": conf_loss,
                "dim_loss": dim_loss,
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
                "train/dim_loss": loss["dim_loss"],
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
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


def orientation_loss(
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
