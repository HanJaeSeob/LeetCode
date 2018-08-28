"""
DCNet trainer
"""
#import pdb
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
import torch.nn.functional as Func
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def l2_loss(x):
    norm = torch.norm(x, 2)
    return norm**2


class PerUttTrainer(object):
    def __init__(self,
                 chimera,
                 alpha=0.1,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 clip_norm=None,
                 num_spks=2):
        self.nnet = chimera
        self.alpha = alpha
        logger.info("chimeraNet:\n{}".format(self.nnet))
        if type(lr) is str:
            lr = float(lr)
            logger.info("Transfrom lr from str to float => {}".format(lr))
        self.optimizer = torch.optim.Adam(
            self.nnet.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, factor=0.5, patience=3)
        self.nnet.to(device)
        self.checkpoint = checkpoint
        self.num_spks = num_spks
        self.clip_norm = clip_norm
        if self.clip_norm:
            logger.info("Clip gradient by 2-norm {}".format(clip_norm))

        if not os.path.exists(self.checkpoint):
            os.makedirs(checkpoint)

    def loss(self, net_embed, net_irm, tgt_index, vad_masks, log_mag, pss_inst, pss_vocal):
        """
        :param net_embed: 128,100*257,20
        :param net_irm: 128,100*257,2
        :param tgt_index: 128,100,257
        :param vad_masks: 128,100,257
        :param mix_spec: 128,100,257
        :param clean_spec:128,100,257
        :return:
        """
        if tgt_index.dim() == 2:
            tgt_index = torch.unsqueeze(tgt_index, 0)
            vad_masks = torch.unsqueeze(vad_masks, 0)

        N, T, F = tgt_index.shape
        vad_masks = vad_masks.view(N, T*F, 1)  # 128,100*257,1

        # encode one-hot
        tgt_embed = torch.zeros([N, T * F, self.num_spks], device=device)
        tgt_embed.scatter_(2, tgt_index.view(N, T * F, 1), 1)  # 128,100*257,2

        # broadcast
        net_embed = net_embed * vad_masks  # 128,100*257,20
        tgt_embed = tgt_embed * vad_masks  # 128,100*257,2

        loss1 = l2_loss(torch.bmm(torch.transpose(net_embed, 1, 2), net_embed)) + \
            l2_loss(torch.bmm(torch.transpose(tgt_embed, 1, 2), tgt_embed)) - \
            l2_loss(torch.bmm(torch.transpose(net_embed, 1, 2), tgt_embed)) * 2

        # loss1 = loss1 / torch.sum(vad_masks)

        loss1 = loss1 / (N*T*F)
        loss1 = loss1 / (T*F) * 100

        """

        def l2_loss(x):
            norm = torch.norm(x, 2)
            return norm ** 2
        v = net_embed * vad_masks  # 128,100*257,20
        y = tgt_embed * vad_masks  # 128,100*257,2
        v_hat = torch.rsqrt(torch.bmm(torch.transpose(v, 1, 2), v))  # 128,20,20
        y_hat = torch.reciprocal(torch.bmm(torch.transpose(y, 1, 2), y))  # 128,2,2
        l_1 = torch.bmm(v, v_hat)  # 128,100*257,20
        l_2 = torch.bmm(torch.bmm(torch.bmm(torch.bmm(y, y_hat), torch.transpose(y, 1, 2)), v), v_hat)  # 128,100*257,20
        loss_DCw = l2_loss(l_1 - l_2)
        """

        net_inst_irm = net_irm[:, :, 0]  # 128,100*257,1
        net_inst_irm = net_inst_irm.view(N, T, F)

        net_vocal_irm = net_irm[:, :, 1]  # 128,100*257,1
        net_vocal_irm = net_vocal_irm.view(N, T, F)

        mix_spec = torch.pow(10, log_mag) - 1e-7

        loss2 = torch.mean((net_inst_irm * mix_spec - pss_inst) ** 2) + \
                torch.mean((net_vocal_irm * mix_spec - pss_vocal) ** 2)

        loss = self.alpha * loss1 + (1 - self.alpha) * loss2

        return loss1, loss2, loss

    def train(self, dataloader):
        """one epoch"""
        self.nnet.train()
        logger.info("Training...")
        tot_loss1 = 0
        tot_loss2 = 0
        tot_loss = 0
        tot_batch = len(dataloader)
        #import pdb;pdb.set_trace()

        for mix_true, inst_true, vocal_true, log_mag, mix_phase, tgt_index, vad_masks, pss_inst, pss_vocal in dataloader:
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                log_mag = log_mag.cuda()
                tgt_index = tgt_index.cuda()
                vad_masks = vad_masks.cuda()
                pss_inst = pss_inst.cuda()
                pss_vocal = pss_vocal.cuda()

            # log_mag = log_mag.requires_grad_()
            tgt_index = tgt_index.requires_grad_()
            vad_masks = vad_masks.requires_grad_()
            pss_inst = pss_inst.requires_grad_()
            pss_vocal = pss_vocal.requires_grad_()

            net_embed, net_irm = self.nnet(log_mag)
            cur_loss1, cur_loss2, cur_loss = self.loss(net_embed, net_irm, tgt_index, vad_masks, log_mag, pss_inst, pss_vocal)
            tot_loss1 += cur_loss1.item()
            tot_loss2 += cur_loss2.item()

            tot_loss += cur_loss.item()
            cur_loss.backward()
            if self.clip_norm:
                nn.utils.clip_grad_norm_(self.nnet.parameters(),
                                         self.clip_norm)
            self.optimizer.step()

        return tot_loss / tot_batch, tot_batch, tot_loss1 / tot_batch, tot_loss2 / tot_batch

    def validate(self, dataloader):
        """one epoch"""
        self.nnet.eval()
        logger.info("Evaluating...")
        tot_loss = 0
        tot_loss1 = 0
        tot_loss2 = 0
        tot_batch = len(dataloader)

        with torch.no_grad():
            for mix_true, inst_true, vocal_true, log_mag, mix_phase, tgt_index, vad_masks, pss_inst, pss_vocal in dataloader:

                if torch.cuda.is_available():
                    log_mag = log_mag.cuda()
                    tgt_index = tgt_index.cuda()
                    vad_masks = vad_masks.cuda()
                    pss_inst = pss_inst.cuda()
                    pss_vocal = pss_vocal.cuda()

                net_embed, net_irm = self.nnet(log_mag)
                cur_loss1, cur_loss2, cur_loss = self.loss(net_embed, net_irm, tgt_index, vad_masks, log_mag, pss_inst, pss_vocal)
                tot_loss += cur_loss.item()

                tot_loss1 += cur_loss1.item()
                tot_loss2 += cur_loss2.item()

        return tot_loss / tot_batch, tot_batch, tot_loss1 / tot_batch, tot_loss2 / tot_batch

    def run(self, train_set, dev_set, num_epoches=20):
        init_loss, _, _, _ = self.validate(dev_set)
        logger.info("Start training for {} epoches".format(num_epoches))
        logger.info("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        torch.save(self.nnet.state_dict(), os.path.join(self.checkpoint, 'chimera++_0.pkl'))
        for epoch in range(1, num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch, train_loss1, train_loss2 = self.train(train_set)
            valid_start = time.time()
            valid_loss, valid_num_batch, valid_loss1, valid_loss2 = self.validate(dev_set)
            valid_end = time.time()
            logger.info(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) | train_loss1 = {:.4e} |"
                " train_loss2 = {:.4e} | dev loss= {:.4e}({:.2f}s/{:d}) | dev_loss1 = {:.4e} |"
                " dev_loss2 = {:.4e}".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, train_loss1, train_loss2,
                    valid_loss, valid_end - valid_start, valid_num_batch,
                    valid_loss1, valid_loss2))

            self.scheduler.step(metrics=valid_loss, epoch=epoch)

            save_path = os.path.join(
                self.checkpoint, "chimera++0.6_{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.nnet.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(num_epoches))







