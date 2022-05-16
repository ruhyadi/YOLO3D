from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from components.kitti_dataset import KITTIDataset

class KITTIDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = '/data/KITTI/training',
        train_sets: str = '/data/KITTI/training/train.txt',
        val_sets: str = '/data/KITTI/training/val.txt',
        batch_size: int = 32,
        
        ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

        # transforms
        # TODO: using albumentations
        self.dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        """ Split dataset to training and validation """
        self.KITTI_train = KITTIDataset(self.hparams.dataset_path, self.hparams.train_sets)
        self.KITTI_val = KITTIDataset(self.hparams.dataset_path, self.hparams.val_sets)
        # TODO: add test datasets dan test sets

    def train_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_train,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.KITTI_val,
            batch_size=self.hparams.batch_size,
            shuffle=False
        )

if __name__ == '__main__':
    print('working')