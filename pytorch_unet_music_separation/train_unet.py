import argparse
from unet import U_Net
from trainer import Trainer
from dataset import SpectrogramDataset
from torch.utils.data import DataLoader
import const as C


def train():
    train_dataset = SpectrogramDataset(C.PATH_FFT)
    valid_dataset = SpectrogramDataset(C.VAL_PATH_FFT)

    train_loader = DataLoader(train_dataset, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

    unet = U_Net()
    trainer = Trainer(unet, C.CHECK_POINT, C.LR)
    trainer.run(train_loader, valid_loader, num_epoches=C.num_epoches)


if __name__ == '__main__':

    train()