import glob
import os

import librosa
import pyworld as pw
import numpy as np
import torch
from tqdm import tqdm

from audio.stft import STFT, TacotronSTFT


def get_pitch(wav, sample_rate, hop_length):
    f0, _ = pw.dio(wav.astype(np.float64), sample_rate,
                   frame_period=hop_length / sample_rate * 1000)
    return f0


def get_mel_energy(wav, filter_length, hop_length, win_length, n_mel_channels, sampling_rate):
    stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels, sampling_rate)
    wav = torch.from_numpy(wav).unsqueeze(0)
    mel, mag = stft.mel_spectrogram(wav)
    mel = mel.squeeze(0)
    mag = mag.squeeze(0)
    energy = torch.norm(mag, dim=0)
    return mel, energy


def pitch_energy_preprocess(preprocess_cfg):
    wav_files = glob.glob(os.path.join(preprocess_cfg.wav_dir, "**", "*.wav"), recursive=True)
    for idx, wavpath in tqdm(enumerate(wav_files), total=len(wav_files)):
        wav, _ = librosa.load(wavpath, sr=preprocess_cfg.sample_rate)
        pitch = get_pitch(wav, preprocess_cfg.sample_rate, preprocess_cfg.hop_length)
        mel, energy = get_mel_energy(wav, preprocess_cfg.filter_length,
                                           preprocess_cfg.hop_length, preprocess_cfg.win_length,
                                           preprocess_cfg.n_mels, preprocess_cfg.sample_rate)

        np.save(f"{preprocess_cfg.pitch_path}/{idx}.npy", pitch[:mel.shape[1]], allow_pickle=False)
        np.save(f"{preprocess_cfg.energy_path}/{idx}.npy", energy.numpy(), allow_pickle=False)
        np.save(f"{preprocess_cfg.mels_path}/{idx}.npy", mel.numpy(), allow_pickle=False)
