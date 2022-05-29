
from pytorch_lightning import LightningDataModule
from components.kitti_dataset import KITTIDataset

class KITTIDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = '../data/KITTI/training',
        train_sets: str = '../data/KITTI/training/train.txt',
        val_sets: str = '../data/KITTI/training/val.txt',
        batch_size: int = 32     
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

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

