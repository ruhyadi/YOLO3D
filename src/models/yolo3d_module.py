"""YOLO3D Multi-head CNN Module for KITTI dataset."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MultiHeadCNN(nn.Module):
    """Multi-head CNN backbone for YOLO3D object detection.

    Predicts:
    - Orientation: sin(θ), cos(θ) for object rotation
    - Location: x, y, z coordinates in 3D space
    - Dimension: height, width, length of 3D bounding box
    - Confidence: objectness score
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 8,  # KITTI has 8 classes
        backbone: str = "resnet",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Backbone network - feature extraction
        if backbone == "resnet":
            self.backbone = self._make_resnet_backbone(input_channels, pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Multi-head outputs
        # Orientation head: sin(θ), cos(θ) for each object
        self.orientation_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 2)  # sin, cos
        )

        # Location head: x, y, z coordinates
        self.location_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 3)  # x, y, z
        )

        # Dimension head: height, width, length
        self.dimension_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 3)  # h, w, l
        )

        # Confidence head: objectness score
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Classification head: object class
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def _make_resnet_backbone(self, input_channels: int, pretrained: bool) -> nn.Module:
        """Create ResNet backbone for feature extraction."""
        import torchvision.models as models

        # Use ResNet-18 as backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer if input channels != 3
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove the final classification layer
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        return backbone

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network."""
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Multi-head predictions
        orientation = self.orientation_head(features)  # [batch, 2]
        location = self.location_head(features)  # [batch, 3]
        dimension = self.dimension_head(features)  # [batch, 3]
        confidence = self.confidence_head(features)  # [batch, 1]
        classification = self.classification_head(features)  # [batch, num_classes]

        return {
            "orientation": orientation,
            "location": location,
            "dimension": dimension,
            "confidence": confidence,
            "classification": classification,
        }


class YOLO3DModule(LightningModule):
    """YOLO3D Lightning Module."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_weights: Dict[str, float] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.net = model

        # Loss weights for multi-task learning
        self.loss_weights = loss_weights or {
            "orientation": 1.0,
            "location": 1.0,
            "dimension": 1.0,
            "confidence": 1.0,
            "classification": 1.0,
        }

        # Loss functions
        self.orientation_criterion = nn.MSELoss()
        self.location_criterion = nn.MSELoss()
        self.dimension_criterion = nn.MSELoss()
        self.confidence_criterion = nn.BCELoss()
        self.classification_criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = Accuracy(task="multiclass", num_classes=8)
        self.val_acc = Accuracy(task="multiclass", num_classes=8)
        self.test_acc = Accuracy(task="multiclass", num_classes=8)

        # For tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the model."""
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, targets = batch

        # Forward pass
        outputs = self.forward(x)

        # Calculate individual losses
        orientation_loss = self.orientation_criterion(
            outputs["orientation"], targets["orientation"]
        )
        location_loss = self.location_criterion(outputs["location"], targets["location"])
        dimension_loss = self.dimension_criterion(outputs["dimension"], targets["dimension"])
        confidence_loss = self.confidence_criterion(
            outputs["confidence"].squeeze(), targets["confidence"].squeeze()
        )
        classification_loss = self.classification_criterion(
            outputs["classification"], targets["class_id"]
        )

        # Combined weighted loss
        loss = (
            self.loss_weights["orientation"] * orientation_loss
            + self.loss_weights["location"] * location_loss
            + self.loss_weights["dimension"] * dimension_loss
            + self.loss_weights["confidence"] * confidence_loss
            + self.loss_weights["classification"] * classification_loss
        )

        preds = torch.argmax(outputs["classification"], dim=1)

        return loss, preds, targets["class_id"]

    def training_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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


if __name__ == "__main__":
    # Test the model
    model = MultiHeadCNN(input_channels=3, num_classes=8)
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
