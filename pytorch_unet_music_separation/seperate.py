#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import utils
import os
import numpy as np
from unet import U_Net

MAX_INT16 = np.iinfo(np.int16).max

"""
Code example for training U-Net
"""

"""
import network

Xlist,Ylist = util.LoadDataset(target="vocal")
print("Dataset loaded.")
network.TrainUNet(Xlist,Ylist,savefile="unet.model",epoch=30)
"""


"""
Code example for performing vocal separation with U-Net
"""

def align(length):
    tmp = 2
    while length > tmp:
        tmp = tmp * 2
    return tmp - length


PATH_test = "test/"
audiolist = os.listdir(PATH_test)

for fname in audiolist:
    mag, phase = utils.LoadAudio(os.path.join(PATH_test, fname))
    leng = mag.shape[1]
    # song's length >= 1024frame
    # song's length = 2^n

    tmp = np.zeros((mag.shape[0], align(leng)), dtype=np.float32)
    mag = np.concatenate((mag, tmp), axis=1)
    print(mag.shape)

    unet = U_Net()

    mask = utils.ComputeMask(mag, unet, "unet_model.pkl", False)
    print(mask.shape)
    mag = mag[:, 0:leng]
    mask = mask[:, 0:leng]

    utils.SaveAudio(
        "enhanced/unet-inst-%s" % fname, mag * mask, phase)
    utils.SaveAudio(
        "enhanced/unet-vocal-%s" % fname, mag * (1 - mask), phase)


"""
fname = "test/A_22_02.wav"
mag, phase = util.LoadAudio(fname)
leng = mag.shape[1]
print(mag.shape)
# song's length >= 1024frame
# song's length = 2^n
tmp = np.zeros((mag.shape[0], align(leng)), dtype=np.float32)
mag = np.concatenate((mag, tmp), axis=1)
print(mag.shape)

mask = util.ComputeMask(mag, unet_model="unet.model", hard=False)
mag = mag[:, 0:leng]
mask = mask[:, 0:leng]

util.SaveAudio(
    "enhanced/unet-vocal", mag*mask, phase)
util.SaveAudio(
    "enhanced/unet-inst", mag*(1-mask), phase)
util.SaveAudio(
    "enhanced/unet-orig", mag, phase)
"""