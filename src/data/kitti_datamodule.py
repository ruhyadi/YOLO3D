"""KITTI Dataset Module for YOLO3D."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class KITTIDataset(Dataset):
    """KITTI 3D Object Detection Dataset."""

    # KITTI classes
    CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            data_dir: Path to KITTI dataset root directory
            split: Either 'training' or 'testing'
            transform: Optional transform to be applied on images
            target_size: Target size for input images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # KITTI directory structure
        self.image_dir = self.data_dir / split / "image_2"
        self.label_dir = self.data_dir / split / "label_2"
        self.calib_dir = self.data_dir / split / "calib"

        # Get all image files
        if self.image_dir.exists():
            self.image_files = sorted(list(self.image_dir.glob("*.png")))
        else:
            self.image_files = []
            print(f"Warning: Image directory {self.image_dir} not found")

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample from the dataset."""
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Load labels (if available)
        label_path = self.label_dir / (image_path.stem + ".txt")

        if label_path.exists() and self.split == "training":
            target = self._parse_label(label_path)
        else:
            # Create dummy target for testing or when labels don't exist
            target = self._create_dummy_target()

        return image, target

    def _parse_label(self, label_path: Path) -> Dict[str, torch.Tensor]:
        """Parse KITTI label file."""
        objects = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return self._create_dummy_target()

        # For simplicity, take the first object (can be extended for multi-object)
        line = lines[0].strip().split()

        if len(line) < 15:
            return self._create_dummy_target()

        # Parse KITTI label format
        class_name = line[0]
        truncated = float(line[1])
        occluded = int(line[2])
        alpha = float(line[3])  # observation angle

        # 2D bounding box
        bbox_2d = [float(line[4]), float(line[5]), float(line[6]), float(line[7])]

        # 3D object dimensions (height, width, length)
        h, w, length = float(line[8]), float(line[9]), float(line[10])

        # 3D object location (x, y, z in camera coordinates)
        x, y, z = float(line[11]), float(line[12]), float(line[13])

        # Rotation around Y-axis in camera coordinates
        ry = float(line[14])

        # Convert to our target format
        class_id = self.CLASS_TO_ID.get(class_name, 0)

        target = {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "orientation": torch.tensor([np.sin(ry), np.cos(ry)], dtype=torch.float32),
            "location": torch.tensor([x, y, z], dtype=torch.float32),
            "dimension": torch.tensor([h, w, length], dtype=torch.float32),
            "confidence": torch.tensor(1.0, dtype=torch.float32),  # Object present
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "bbox_2d": torch.tensor(bbox_2d, dtype=torch.float32),
        }

        return target

    def _create_dummy_target(self) -> Dict[str, torch.Tensor]:
        """Create dummy target when labels are not available."""
        return {
            "class_id": torch.tensor(0, dtype=torch.long),
            "orientation": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "location": torch.tensor([0.0, 0.0, 10.0], dtype=torch.float32),
            "dimension": torch.tensor([1.5, 1.6, 3.9], dtype=torch.float32),
            "confidence": torch.tensor(0.0, dtype=torch.float32),  # No object
            "alpha": torch.tensor(0.0, dtype=torch.float32),
            "bbox_2d": torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=torch.float32),
        }


class KITTIDataModule(LightningDataModule):
    """KITTI Data Module for Lightning."""

    def __init__(
        self,
        data_dir: str = "data/KITTI",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.2, 0.1),
        target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Data transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        """Get the number of classes."""
        return len(KITTIDataset.CLASSES)

    def prepare_data(self) -> None:
        """Download data if needed."""
        # KITTI dataset needs to be downloaded manually
        # Just check if the directory exists
        data_path = Path(self.hparams.data_dir)
        if not data_path.exists():
            print(f"Warning: KITTI data directory {data_path} does not exist.")
            print(
                "Please download KITTI dataset from: http://www.cvlibs.net/datasets/kitti/eval_object.php"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # Check if data directory exists
        if not Path(self.hparams.data_dir).exists():
            print("Creating dummy dataset for testing...")
            # Create dummy datasets for development
            self.data_train = self._create_dummy_dataset(100)
            self.data_val = self._create_dummy_dataset(20)
            self.data_test = self._create_dummy_dataset(10)
            return

        if not self.data_train and not self.data_val and not self.data_test:
            # Create full dataset
            dataset = KITTIDataset(
                data_dir=self.hparams.data_dir,
                split="training",
                transform=None,  # We'll set transforms later
            )

            # Split dataset
            train_size = int(self.hparams.train_val_test_split[0] * len(dataset))
            val_size = int(self.hparams.train_val_test_split[1] * len(dataset))
            test_size = len(dataset) - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

            # Set transforms
            self.data_train.dataset.transform = self.train_transforms
            self.data_val.dataset.transform = self.val_transforms
            self.data_test.dataset.transform = self.val_transforms

    def _create_dummy_dataset(self, size: int) -> Dataset:
        """Create dummy dataset for testing when KITTI data is not available."""

        class DummyDataset(Dataset):
            def __init__(self, size: int, transform=None):
                self.size = size
                self.transform = transform or transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Random dummy image
                image = torch.randn(3, 224, 224)
                # Random dummy target
                target = {
                    "class_id": torch.randint(0, 8, (1,)).long().squeeze(),
                    "orientation": torch.randn(2),
                    "location": torch.randn(3),
                    "dimension": torch.abs(torch.randn(3)) + 0.5,
                    "confidence": torch.rand(1).squeeze(),
                    "alpha": torch.randn(1).squeeze(),
                    "bbox_2d": torch.abs(torch.randn(4)) * 100,
                }
                return image, target

        return DummyDataset(size)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    # Test the dataset
    dataset = KITTIDataset("data/KITTI", "training")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target keys: {list(target.keys())}")

    # Test data module
    datamodule = KITTIDataModule()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    images, targets = batch

    print(f"Batch image shape: {images.shape}")
    print(f"Batch target class_id shape: {targets['class_id'].shape}")
