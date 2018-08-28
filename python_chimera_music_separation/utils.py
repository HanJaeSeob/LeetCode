import os
import yaml
import numpy as np
import librosa as audio_lib
from librosa.core import stft, load, istft, resample
import config as C


def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find config file...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)

    num_frames = config_dict["spectrogram_reader"]["frame_length"]
    num_bins = num_frames // 2 + 1
    return num_bins, config_dict


def apply_cmvn(x):
    return (x - x.mean()) / x.std()


def unapply_cmvn(x):
    return x * x.std() + x.mean()


def SaveSpectrogram(y_mix, y_vocal, y_inst, fname, original_sr=44100):
    """get data, labels and save every (100,257) npz"""
    y_mix = resample(y_mix, original_sr, C.SR)
    y_vocal = resample(y_vocal, original_sr, C.SR)
    y_inst = resample(y_inst, original_sr, C.SR)

    i = 0
    cnt = 0
    # per second a unit
    while i + C.SR <= len(y_mix):
        T_mix = y_mix[i:i+C.SR]
        T_vocal = y_vocal[i:i+C.SR]
        T_inst = y_inst[i:i+C.SR]

        spec_mix = stft(T_mix, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
        spec_vocal = stft(T_vocal, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
        spec_inst = stft(T_inst, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)

        S_mix = np.abs(spec_mix).astype(np.float32)
        S_vocal = np.abs(spec_vocal).astype(np.float32)
        S_inst = np.abs(spec_inst).astype(np.float32)

        P_mix = np.angle(spec_mix)
        P_vocal = np.angle(spec_vocal)
        P_inst = np.angle(spec_inst)

        """
        norm = S_mix.max()
        if norm == 0:
            print("####")
        S_mix /= norm
        S_vocal /= norm
        S_inst /= norm
        """

        log_mix = np.log10(S_mix + 1e-7)

        # Create IBM (126, 257)
        specs = []
        specs.append(S_vocal)
        specs.append(S_inst)
        specs = np.array(specs)
        Y = np.argmax(specs, axis=0)

        # Create mask silence components
        spectra = np.exp(S_mix)
        spectra_db = 20 * np.log10(spectra)
        max_magnitude_db = np.max(spectra_db)
        threshold = 10 ** ((max_magnitude_db - C.DB_THRESHOLD) / 20)
        vad_mask = np.array(spectra > threshold, dtype=np.float32)

        pss_inst = (S_inst * np.cos(P_mix - P_inst)).astype(np.float32)
        pss_inst = np.maximum(pss_inst, 0)
        pss_inst = np.minimum(pss_inst, S_mix)

        pss_vocal = (S_vocal * np.cos(P_mix - P_vocal)).astype(np.float32)
        pss_vocal = np.maximum(pss_vocal, 0)
        pss_vocal = np.minimum(pss_vocal, S_mix)

        np.savez(os.path.join(C.PATH_FFT, fname + str(cnt) + ".npz"),
                 mix_true=T_mix, inst_true=T_inst, vocal_true=T_vocal,
                 log_mag=log_mix, mix_phase=P_mix, ibm_label=Y,
                 vad_label=vad_mask, pss_inst=pss_inst, pss_vocal=pss_vocal)

        i += C.SR // 2
        cnt += 1
