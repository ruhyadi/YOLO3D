"""
KITTI datamodule for MultiBin model inherited from pytorch LightningDataModule.
"""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.kitti_dataloader import KittiDataloader


class KittiDatamodule(LightningDataModule):
    """
    KITTI datamodule for MultiBin model inherited from pytorch LightningDataModule.
    """

    def __init__(
        self,
        images_dir: str = "data/kitti/image_2",
        labels_dir: str = "data/kitti/label_2",
        calib_path: str = "assets/kitti_calib.txt",
        train_sets_path: str = "assets/kitti_train.txt",
        val_sets_path: str = "assets/kitti_val.txt",
        classes: List[str] = ["Car", "Pedestrian", "Cyclist"],
        n_bins: int = 2,
        overlap: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Initialize KITTI datamodule."""
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.calib_path = Path(calib_path)
        self.train_sets_path = Path(train_sets_path)
        self.val_sets_path = Path(val_sets_path)
        self.classes = classes
        self.n_bins = n_bins
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`

        Args:
            stage (str, optional): 'fit' or 'test'.
        """
        if not self.data_train and not self.data_val:
            self.data_train = KittiDataloader(
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                calib_path=self.calib_path,
                sets_path=self.train_sets_path,
                classes=self.classes,
                n_bins=self.n_bins,
                overlap=self.overlap,
            )
            self.data_val = KittiDataloader(
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                calib_path=self.calib_path,
                sets_path=self.val_sets_path,
                classes=self.classes,
                n_bins=self.n_bins,
                overlap=self.overlap,
            )

    def train_dataloader(self) -> DataLoader:
        """Load train dataloader."""

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Load validation dataloader."""

        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
