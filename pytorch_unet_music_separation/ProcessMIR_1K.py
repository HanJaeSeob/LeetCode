#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:49:54 2017

@author: wuyiming
"""

import numpy as np
from librosa.util import find_files
from librosa.core import load
import os.path
import utils

music_path = "/media/data/rainiejjli/pytorch_UNet_DSD/MIR-1K/music/"
vocal_path = "/media/data/rainiejjli/pytorch_UNet_DSD/MIR-1K/vocal/"

musiclist = find_files(music_path, ext="wav")
musiclist.sort()
vocallist = find_files(music_path, ext="wav")
vocallist.sort()

print(len(musiclist))

def make_same_length(speech,len_ref):
    len_s = len(speech)
    rep = int(np.floor(len_ref / len_s))
    left = len_ref - len_s * rep
    temp_speech = np.tile(speech, [1, rep])
    temp_speech.shape = (temp_speech.shape[1],)
    speech = np.hstack((temp_speech, speech[:left]))
    return speech
"""
for idx in range(len(musiclist)):
    fname1 = os.path.split(musiclist[idx])[-1]
    fname2 = os.path.split(vocallist[idx])[-1]


    print("Processing: %s" % fname1)
    print("--%s" % fname2)
    music, _ = load(musiclist[idx], sr=16000)
    vocal, _ = load(vocallist[idx], sr=16000)
    if len(music) < 1024 * 176:
        music = make_same_length(music, 1024 * 176)
    if len(vocal) < 1024 * 176:
        vocal = make_same_length(vocal, 1024 * 176)
    mix = music + vocal
    utils.SaveSpectrogram(mix, vocal, music, fname1)


print("Constructing random mix...")

rand = np.random.randint(len(musiclist), size=1000)

count = 1

for i in range(0, 1000, 2):
    y1, _ = load(musiclist[rand[i]], sr=16000)
    y2, _ = load(vocallist[rand[i+1]], sr=16000)

    if len(y1) < 1024 * 176:
        y1 = make_same_length(y1, 1024 * 176)
    if len(y2) < 1024 * 176:
        y2 = make_same_length(y2, 1024 * 176)

    fname1 = os.path.split(musiclist[rand[i]])[-1]
    fname2 = os.path.split(vocallist[rand[i+1]])[-1]

    print("Processing: %s" % fname1)
    print("--%s" % fname2)

    minlen = min(len(y1), len(y2))
    inst = y1[:minlen]
    vocal = y2[:minlen]
    mix = inst+vocal

    fname = "mir-1k_random%02d" % count
    utils.SaveSpectrogram(mix, vocal, inst, fname)
    count += 1
    print("Saved %s.npz" % fname)

"""
arr = np.arange(1000)
np.random.shuffle(arr)
count = 1
for i in range(500):
    y1, _ = load(musiclist[i], sr=16000)
    y2, _ = load(vocallist[arr[i]], sr=16000)

    if len(y1) < 1024*176:
        y1 = make_same_length(y1, 1024*176)
    if len(y2) < 1024*176:
        y2 = make_same_length(y2, 1024*176)

    fname1 = os.path.split(musiclist[i])[-1]
    fname2 = os.path.split(vocallist[arr[i]])[-1]

    print("Processing: %s" % fname1)
    print("--%s" % fname2)

    minlen = min(len(y1), len(y2))
    inst = y1[:minlen]
    vocal = y2[:minlen]
    mix = inst+vocal

    fname = "mir-1k_shuffle%02d" % count
    utils.SaveSpectrogram(mix, vocal, inst, fname)
    count += 1
    print("Saved %s.npz" % fname)
