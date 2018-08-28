# !/usr/bin/env python
# coding=utf-8

import argparse
import os
import sklearn
import numpy as np

np.set_printoptions(threshold=np.inf)
import torch as th
import scipy.io as sio

from utils import parse_yaml
from chimera import chimeraNet
from librosa.core import load, stft, istft
from librosa.output import write_wav
import config as C
from torch.autograd import Variable
import soundfile as sf

MAX_INT16 = np.iinfo(np.int16).max


def SaveAudio(fname, mag, phase):
    y = istft(mag*phase, win_length=C.FFT_SIZE, hop_length=C.H)
    write_wav(fname, (y / np.max(np.abs(y)) * MAX_INT16).astype(np.int16), C.SR)

class DeepCluster(object):
    def __init__(self, dcnet, dcnet_state):
        if not os.path.exists(dcnet_state):
            raise RuntimeError(
                "Could not find state file {}".format(dcnet_state))
        self.dcnet = dcnet

        self.dcnet.load_state_dict(th.load(dcnet_state))
        self.dcnet.cuda()
        self.dcnet.eval()
        # self.kmeans = sklearn.cluster.KMeans(n_clusters=2)

    def separate(self, spectra):
        """
        Arguments
            spectra:    log-magnitude spectrogram(real numbers)
            vad_mask:   binary mask for non-silence bins(if non-sil: 1)
            return
                spk_masks: binary masks for each speaker
        """
        # TF x D
        net_embed, net_irm = self.dcnet(Variable(th.from_numpy(spectra).cuda()))
        # net_embed = net_embed.cpu().data.numpy()
        net_irm = net_irm.cpu().data.numpy()
        # print(net_embed.shape)
        # net_embed = net_embed.reshape(-1, net_embed.shape[-1])

        # eg = self.kmeans.fit_predict(net_embed)
        # print(eg.shape)

        # imgs = np.zeros((2, eg.size))
        # for i in range(2):
        #    imgs[i, eg == i] = 1

        return net_irm



def run(args):
    PATH_test = "test/"
    audiolist = os.listdir(PATH_test)
    # test_wav = args.test_wav
    num_bins, config_dict = parse_yaml("train.yaml")

    dcnet = chimeraNet(num_bins, **config_dict["dcnet"])

    frame_length = config_dict["spectrogram_reader"]["frame_length"]
    frame_shift = config_dict["spectrogram_reader"]["frame_shift"]

    cluster = DeepCluster(
        dcnet,
        args.dcnet_state)

    for fname in audiolist:

        sig, _ = load(os.path.join(PATH_test, fname), sr=C.SR)
        i = 0
        input = []
        mix_S = []
        mix_P = []
        while i + C.SR <= len(sig):
            mix_ = stft(sig[i:i+C.SR], frame_length, frame_shift).transpose()
            mix_spec = np.abs(mix_).astype(np.float32)
            phase = np.exp(1.j*np.angle(mix_))
            log_mix = np.log10(mix_spec + 1e-7)
            input.append(log_mix)
            mix_S.append(mix_spec)
            mix_P.append(phase)
            i += C.SR
        input = np.array(input)
        mix_S = np.array(mix_S).reshape(-1, 257)
        mix_P = np.array(mix_P).reshape(-1, 257)

        net_irm= cluster.separate(input)

        """
        i = 0
        for img in imgs:
            mask = img.reshape(-1, 257)
            print(mask.shape)
            mag = (mix_S * mask).transpose()
            y = istft(mag * mix_P.transpose(), hop_length=frame_shift, win_length=frame_length)
            write_wav('DC_{}.wav'.format(i), (y / np.max(np.abs(y)) * MAX_INT16).astype(np.int16), C.SR)
            i += 1
        """


        mask1 = net_irm[:, :, 0].reshape(-1, 257)
        mag1 = (mix_S * mask1).transpose()
        y1 = istft(mag1 * mix_P.transpose(), hop_length=frame_shift, win_length=frame_length)

        mask2 = net_irm[:, :, 1].reshape(-1, 257)
        mag2 = (mix_S * mask2).transpose()
        y2 = istft(mag2 * mix_P.transpose(), hop_length=frame_shift, win_length=frame_length)

        for i in range(5):
            if len(sig) > len(y1):
                length = len(y1)
            else:
                length = len(sig)
            y1 = y1[: length]
            y2 = y2[: length]
            sig = sig[: length]
            delta = sig - (y1 + y2)
            y1 = y1 + delta / 2
            y2 = y2 + delta / 2
            s1_mix_ = stft(y1, frame_length, frame_shift).transpose()
            phase1 = np.exp(1.j * np.angle(s1_mix_))
            s2_mix_ = stft(y2, frame_length, frame_shift).transpose()
            phase2 = np.exp(1.j * np.angle(s2_mix_))
            mag1 = mag1.transpose()[: len(phase1)].transpose()
            mag2 = mag2.transpose()[: len(phase2)].transpose()
            y1 = istft(mag1* phase1.transpose(), hop_length=frame_shift, win_length=frame_length)
            y2 = istft(mag2 * phase2.transpose(), hop_length=frame_shift, win_length=frame_length)
        """
        sig = sig[: len(y1)]
        delta = sig - (y1 + y2)
        y1 = y1 + delta / 2
        y2 = y2 + delta / 2
        s1_mix_ = stft(y1, frame_length, frame_shift).transpose()
        phase1 = np.exp(1.j * np.angle(s1_mix_))
        s2_mix_ = stft(y2, frame_length, frame_shift).transpose()
        phase2 = np.exp(1.j * np.angle(s2_mix_))
        mag1 = mag1.transpose()[: len(phase1)].transpose()
        mag2 = mag2.transpose()[: len(phase2)].transpose()
        y1 = istft(mag1 * phase1.transpose(), hop_length=frame_shift, win_length=frame_length)
        y2 = istft(mag2 * phase2.transpose(), hop_length=frame_shift, win_length=frame_length)
        """
        write_wav('enhanced/chimera-inst-%s' % fname, (y1 / np.max(np.abs(y1)) * MAX_INT16).astype(np.int16), C.SR)
        write_wav('enhanced/chimera-vocal-%s' % fname, (y2 / np.max(np.abs(y2)) * MAX_INT16).astype(np.int16), C.SR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "music separation using masks clustered on embeddings of DCNet"
    )
    parser.add_argument(
        "--test-wave",
        default="linjunjie_song.wav",
        type=str,
        dest="test_wav",
        help="test wave name")
    parser.add_argument(
        "--dcnet-state",
        default="models/chimera_model.pkl",
        type=str,
        dest="dcnet_state",
        help="model path")
    args = parser.parse_args()
    run(args)
