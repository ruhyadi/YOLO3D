"""
Script for training Regressor Model with pytorch lightning
"""

import argparse
from asyncio.log import logger
import os
import sys
from pathlib import Path

from script.Dataset_lightning import Dataset, KITTIDataModule
from script.Model_lightning import Model

from comet_ml import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def train(
    train_path=ROOT / 'dataset/KITTI/training',
    checkpoint_path=ROOT / 'weights/checkpoints',
    model_select='resnet18',
    epochs=10,
    batch_size=32,
    num_workers=2,
    gpu=1,
    val_split=0.1,
    model_path=ROOT / 'weights/',
    api_key=''
    ):

    # comet ml
    comet_logger = CometLogger(
        api_key=api_key,
        project_name="YOLO3D"
        )

    # initiate callback mode
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename='model_{epoch:02d}_{val_loss:.2f}',
        save_top_k=3,
        mode='min'
        )
    
    # initiate trainer
    trainer = Trainer(
        logger=comet_logger,
        callbacks=[checkpoint_callback],
        gpus=gpu,
        min_epochs=1,
        max_epochs=epochs)

    # initiate model
    model = Model(model_select=model_select)

    # load weights
    try:
        latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
    except:
        latest_model = None
    if latest_model is not None :
        model.load_from_checkpoint(latest_model)

        print(f'[INFO] Use previous model {latest_model}')

    # initiate dataset
    dataset = KITTIDataModule(
        dataset_path=train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
    )

    # train model
    trainer.fit(model=model, datamodule=dataset)

def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--train_path', type=str, default=ROOT / 'dataset_dummy/training', help='Training path KITTI')
    parser.add_argument('--checkpoint_path', type=str, default=ROOT / 'weights/checkpoint', help='Checkpoint directory')
    parser.add_argument('--model_select', type=str, default='resnet18', help='Model selection: {resnet18, vgg11}')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Total # workers, for colab & kaggle use 2')
    parser.add_argument('--gpu', type=int, default=0, help='Numbers of GPU, default=1')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split percentage')
    parser.add_argument('--model_path', type=str, default=ROOT / 'weights', help='Weights path, for load and save model')
    parser.add_argument('--api_key', type=str, default='', help='API key for comet.ml')

    opt = parser.parse_args()

    return opt

def main(opt):
    train(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


