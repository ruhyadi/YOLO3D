"""
Script for training Regressor Model
"""

import argparse
import os
from random import shuffle
import sys
from pathlib import Path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from comet_ml import Experiment

from script.Dataset import Dataset
from script.Model import ResNet18, VGG11, OrientationLoss

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, vgg11
from torch.utils import data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# model factory to choose model
model_factory = {
    'resnet18': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet18': ResNet18,
    'vgg11': VGG11
}


def train(
    epochs=10,
    batch_size=32,
    alpha=0.6,
    w=0.4,
    num_workers=2,
    lr=0.0001,
    save_epoch=10,
    train_path=ROOT / 'dataset/KITTI/training',
    model_path=ROOT / 'weights/',
    select_model='resnet18',
    api_key=''
    ):

    # directory
    train_path = str(train_path)
    model_path = str(model_path)

    # dataset
    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path)

    # hyper_params
    hyper_params = {
    'epochs': epochs,
    'batch_size': batch_size,
    'w': w,
    'num_workers': num_workers,
    'lr': lr,
    'shuffle': True
    }

    # comet ml experiment
    experiment = Experiment(api_key, project_name="YOLO3D")
    experiment.log_parameters(hyper_params)

    # data generator
    data_gen = data.DataLoader(
        dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=hyper_params['shuffle'],
        num_workers=hyper_params['num_workers'])

    # model
    base_model = model_factory[select_model]
    model = regressor_factory[select_model](model=base_model).cuda()
    
    # optimizer
    opt_SGD = torch.optim.SGD(model.parameters(), lr=hyper_params['lr'], momentum=0.9)

    # loss function
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load previous weights
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'[INFO] Using previous model {latest_model} at {first_epoch} epochs')
        print('[INFO] Resuming training...')

    total_num_batches = int(len(dataset) / hyper_params['batch_size'])

    with experiment.train():
        for epoch in range(first_epoch, int(hyper_params['epochs'])+1):
            curr_batch = 0
            passes = 0
            with tqdm(data_gen, unit='batch') as tepoch:
                for local_batch, local_labels in tepoch:
                    # progress bar
                    tepoch.set_description(f'Epoch {epoch}')

                    # ground-truth
                    truth_orient = local_labels['Orientation'].float().cuda()
                    truth_conf = local_labels['Confidence'].float().cuda()
                    truth_dim = local_labels['Dimensions'].float().cuda()

                    # convert to cuda
                    local_batch = local_batch.float().cuda()

                    # forward
                    [orient, conf, dim] = model(local_batch)

                    # loss
                    orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
                    dim_loss = dim_loss_func(dim, truth_dim)
                    
                    truth_conf = torch.max(truth_conf, dim=1)[1]
                    conf_loss = conf_loss_func(conf, truth_conf)

                    loss_theta = conf_loss + w*orient_loss
                    loss = alpha*dim_loss + loss_theta

                    # write tensorboard and comet ml
                    writer.add_scalar('Loss/train', loss, epoch)
                    experiment.log_metric('Loss/train', loss, epoch=epoch)

                    opt_SGD.zero_grad()
                    loss.backward()
                    opt_SGD.step()

                    # progress bar update
                    tepoch.set_postfix(loss=loss.item())

            if epoch % save_epoch == 0:
                model_name = os.path.join(model_path, f'{select_model}_epoch_{epoch}.pkl')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                }, model_name)
                print(f'[INFO] Saving weights as {model_name}')

    writer.flush()
    writer.close()

def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size')
    parser.add_argument('--alpha', type=float, default=0.6, help='Aplha default=0.6 DONT CHANGE')
    parser.add_argument('--w', type=float, default=0.4, help='w DONT CHANGE')
    parser.add_argument('--num_workers', type=int, default=2, help='Total # workers, for colab & kaggle use 2')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_epoch', type=int, default=10, help='Save model every # epochs')
    parser.add_argument('--train_path', type=str, default=ROOT / 'dataset/KITTI/training', help='Training path KITTI')
    parser.add_argument('--model_path', type=str, default=ROOT / 'weights', help='Weights path, for load and save model')
    parser.add_argument('--select_model', type=str, default='resnet18', help='Model selection: {resnet18, vgg11}')
    parser.add_argument('--api_key', type=str, default='', help='API key for comet.ml')

    opt = parser.parse_args()

    return opt

def main(opt):
    train(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


