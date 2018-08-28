import os
import logging
import numpy as np
import torch as th
import random
from librosa.util import find_files
from torch.utils import data
import config as C


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class SpectrogramDataset(data.Dataset):
    def __init__(self, file_path):
        self.filelist_fft = find_files(file_path, "npz")

    def __getitem__(self, idx):
        dat = np.load(self.filelist_fft[idx])
        mix_true = dat["mix_true"]
        inst_true = dat["inst_true"]
        vocal_true = dat["vocal_true"]
        log_mag = dat["mix_spec"]
        mix_phase = dat["mix_phase"]
        ibm_label = dat["ibm_label"]
        vad_label = dat["vad_label"]
        pss_inst = dat["pss_inst"]
        pss_vocal = dat["pss_vocal"]

        return th.from_numpy(mix_true), th.from_numpy(inst_true), th.from_numpy(vocal_true), \
               th.from_numpy(log_mag), th.from_numpy(mix_phase), \
               th.from_numpy(ibm_label), th.from_numpy(vad_label), th.from_numpy(pss_inst), th.from_numpy(pss_vocal)


    def __len__(self):
        return len(self.filelist_fft)













