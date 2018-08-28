import pdb

import argparse
import time
from utils import parse_yaml
from chimera import chimeraNet
from trainer import PerUttTrainer
from dataset import SpectrogramDataset
from torch.utils.data import DataLoader


def train(args):
    num_bins, config_dict = parse_yaml(args.config)
    # reader_conf = config_dict["spectrogram_reader"]
    loader_conf = config_dict["dataloader"]
    dcnet_conf = config_dict["dcnet"]
    train_config = config_dict["trainer"]

    train_dataset = SpectrogramDataset(loader_conf["train_path_npz"])
    valid_dataset = SpectrogramDataset(loader_conf["valid_path_npz"])

    train_loader = DataLoader(train_dataset, batch_size=loader_conf["batch_size"], shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=loader_conf["batch_size"], shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

    chimera = chimeraNet(num_bins, **dcnet_conf)
    trainer = PerUttTrainer(chimera, args.alpha, **train_config)
    trainer.run(train_loader, valid_loader, num_epoches=args.num_epoches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to train chimeraNet, configured by .yaml files")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        dest="alpha",
        help="weight of loss")
    args = parser.parse_args()

    parser.add_argument(
        "--num_epoches",
        type=int,
        default=200,
        dest="num_epoches",
        help="Number of epoches to train dcnet")
    args = parser.parse_args()
    train(args)