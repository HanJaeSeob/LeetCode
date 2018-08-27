#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from librosa.util import find_files
from librosa.core import stft, load, istft, resample
from librosa.output import write_wav
import const as C
import numpy as np
from chainer import config
import os.path

MAX_INT16 = np.iinfo(np.int16).max


def SaveSpectrogram(y_mix, y_vocal, y_inst, fname, original_sr=44100):
    """extract features and save"""
    y_mix = resample(y_mix, original_sr, C.SR)
    y_vocal = resample(y_vocal, original_sr, C.SR)
    y_inst = resample(y_inst, original_sr, C.SR)

    S_mix = np.abs(
        stft(y_mix, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_vocal = np.abs(
        stft(y_vocal, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)

    norm = S_mix.max()
    S_mix /= norm
    S_vocal /= norm
    S_inst /= norm

    # np.savez(os.path.join(C.PATH_FFT, fname+".npz"), mix=S_mix, vocal=S_vocal, inst=S_inst)

    # Generate sequence (1,512,128) and save
    cnt = 1
    i = 0
    while i + C.PATCH_LENGTH < S_mix.shape[1]:
        mix_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        #vocal_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        inst_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        mix_spec[0, :, :] = S_mix[1:, i:i+C.PATCH_LENGTH]
        #vocal_spec[0, :, :] = S_vocal[1:, i:i + C.PATCH_LENGTH]
        inst_spec[0, :, :] = S_inst[1:, i:i + C.PATCH_LENGTH]

        np.savez(os.path.join(C.VAL_PATH_FFT, fname + str(cnt) + ".npz"),
                 data=mix_spec, label=inst_spec)

        i += C.PATCH_LENGTH
        cnt += 1

    if S_mix.shape[1] >= 128:
        mix_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        #vocal_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        inst_spec = np.zeros((1, 512, C.PATCH_LENGTH), dtype=np.float32)
        mix_spec[0, :, :] = S_mix[1:, S_mix.shape[1]-C.PATCH_LENGTH:S_mix.shape[1]]
        #vocal_spec[0, :, :] = S_vocal[1:, S_mix.shape[1] - C.PATCH_LENGTH:S_mix.shape[1]]
        inst_spec[0, :, :] = S_inst[1:, S_mix.shape[1]-C.PATCH_LENGTH:S_mix.shape[1]]

        np.savez(os.path.join(C.VAL_PATH_FFT, fname + str(cnt) + ".npz"),
                 data=mix_spec, label=inst_spec)
        cnt += 1


def LoadDataset(target="vocal", path=""):
    filelist_fft = find_files(path, ext="npz")
    Xlist = []
    Ylist = []
    for file_fft in filelist_fft:
        dat = np.load(file_fft)
        Xlist.append(dat["mix"])
        if target == "vocal":
            assert(dat["mix"].shape == dat["vocal"].shape)
            Ylist.append(dat["vocal"])
        else:
            assert(dat["mix"].shape == dat["inst"].shape)
            Ylist.append(dat["inst"])
    return Xlist, Ylist


def LoadAudio(fname):
    y, sr = load(fname, sr=C.SR)
    spec = stft(y, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase


def SaveAudio(fname, mag, phase):
    y = istft(mag*phase, hop_length=C.H, win_length=C.FFT_SIZE)
    write_wav(fname, (y / np.max(np.abs(y)) * MAX_INT16).astype(np.int16), C.SR, norm=True)


def ComputeMask(input_mag, unet, unet_params, hard=True):
    # unet = network_pytorch.U_Net()

    unet.load_state_dict(torch.load(unet_params))
    unet.cuda()
    unet.eval()

    input_mag = torch.FloatTensor(input_mag[np.newaxis, np.newaxis, 1:, :])
    input_mag = input_mag.cuda()
    input_mag = Variable(input_mag)

    mask = unet(input_mag).cpu().data.numpy()[0, 0, :, :]
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    if hard:
        hard_mask = np.zeros(mask.shape, dtype="float32")
        hard_mask[mask > 0.5] = 1
        return hard_mask
    else:
        return mask


def phase_MISI(inst_esti, vocal_esti, mix):
    delta = mix - (inst_esti + vocal_esti)
    inst = inst_esti + delta / 2
    vocal = vocal_esti + delta / 2
    S_inst = stft(inst, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    S_vocal = stft(vocal, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    P_inst = np.exp(1.j * np.angle(S_inst))
    P_vocal = np.exp(1.j * np.angle(S_vocal))
    return P_inst, P_vocal