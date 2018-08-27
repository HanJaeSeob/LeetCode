import os
import logging
import numpy as np
import torch
import random
from librosa.util import find_files
from torch.utils import data
import const as C


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
        data = dat["data"]
        label = dat["label"]

        return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.filelist_fft)