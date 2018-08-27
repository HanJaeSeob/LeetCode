"""
UNet trainer
"""
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, unet, checkpoint="checkpoint", lr=1e-5):
        self.nnet = unet
        # self.nnet.load_state_dict(torch.load("models_MIR_1k/unet_34_trainloss_7.9405e-04_valloss_8.2840e-04.pkl"))
        logger.info("UNet:\n{}".format(self.nnet))
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)
        self.nnet.to(device)
        self.checkpoint = checkpoint

        if not os.path.exists(self.checkpoint):
            os.makedirs(checkpoint)

    def loss(self, estimated_mask, mix_spec, inst_spec):
        return F.l1_loss(mix_spec * estimated_mask, inst_spec)

    def train(self, dataloader):
        """one epoch"""
        self.nnet.train()
        logger.info("Training...")
        tot_batch = len(dataloader)
        tot_loss = 0
        for mix_spec, inst_spec in dataloader:
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                mix_spec = mix_spec.cuda()
                inst_spec = inst_spec.cuda()

            mix_spec = Variable(mix_spec, requires_grad=True)
            inst_spec = Variable(inst_spec, requires_grad=False)

            estimated_mask = self.nnet(mix_spec)
            cur_loss = self.loss(estimated_mask, mix_spec, inst_spec)
            tot_loss += cur_loss.item()
            cur_loss.backward()
            self.optimizer.step()
        return tot_loss / tot_batch, tot_batch

    def validate(self, dataloader):
        """one epoch"""
        self.nnet.eval()
        logger.info("Evaluating...")
        tot_loss = 0
        tot_batch = len(dataloader)
        with torch.no_grad():

            for mix_spec, inst_spec in dataloader:
                if torch.cuda.is_available():
                    mix_spec = mix_spec.cuda()
                    inst_spec = inst_spec.cuda()

                mix_spec = Variable(mix_spec)
                inst_spec = Variable(inst_spec)

                estimated_mask = self.nnet(mix_spec)
                cur_loss = self.loss(estimated_mask, mix_spec, inst_spec)
                tot_loss += cur_loss.item()
        return tot_loss / tot_batch, tot_batch

    def run(self, train_set, dev_set, num_epoches=20):
        init_loss, _ = self.validate(dev_set)
        logger.info("Start training for {} epoches".format(num_epoches))
        logger.info("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        torch.save(self.nnet.state_dict(), os.path.join(self.checkpoint, 'unet_0.pkl'))
        for epoch in range(1, num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch = self.train(train_set)
            valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set)
            valid_end = time.time()
            logger.info(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) |"
                " dev loss= {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, valid_loss, valid_end - valid_start,
                    valid_num_batch))
            self.scheduler.step(metrics=valid_loss, epoch=epoch)
            save_path = os.path.join(
                self.checkpoint, "unet{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.nnet.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(num_epoches))







