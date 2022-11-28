import glob

import numpy as np
import os

import yaml
from tqdm import tqdm

from utils import Dict
from sklearn.preprocessing import StandardScaler


def get_files(path, extension=".wav"):
    return list(glob.iglob(f"{path}/**/*{extension}", recursive=True))


def compute_mean_std(files, refresh=False):
    scaler = StandardScaler()
    for file in tqdm(files):
        entity = np.load(file)
        q25 = np.quantile(entity, 0.25)
        q75 = np.quantile(entity, 0.75)
        l_bound = q25 - 1.5 * (q75 - q25)
        u_bound = q75 + 1.5 * (q75 - q25)
        entity[(entity < l_bound) | (entity > u_bound)] = 0
        scaler.partial_fit(entity.reshape(-1, 1))
        if refresh:
            np.save(file, entity, allow_pickle=False)
    return scaler.mean_, scaler.scale_

def compute_min_max(files, mean, std):
    min_val = np.finfo(np.float32).max
    max_val = np.finfo(np.float32).min
    for file in tqdm(files):
        entity = np.load(file)
        entity = (entity - mean) / std
        max_val = max(max_val, max(entity))
        min_val = min(min_val, min(entity))
    return min_val, max_val

if __name__ == "__main__":

    cfg = yaml.load(open("../config/fastspeech2.yaml", "r"), Loader=yaml.FullLoader)
    cfg = Dict(cfg)

    energy_path = os.path.join("../", cfg.data.data_dir, "energy")
    pitch_path = os.path.join("../", cfg.data.data_dir, "pitch")

    energy_files = get_files(energy_path, extension=".npy")
    pitch_files = get_files(pitch_path, extension=".npy")

    pitch_mean, pitch_std = compute_mean_std(pitch_files)

    print(f"Mean pitch : {pitch_mean}")
    print(f"Std pitch : {pitch_std}")

    min_p, max_p = compute_min_max(pitch_files, pitch_mean, pitch_std)

    print(f"Min Pitch : {min_p}")
    print(f"Max Pitch : {max_p}")

    energy_mean, energy_std = compute_mean_std(energy_files)

    print(f"Mean energy : {energy_mean}")
    print(f"Std energy : {energy_std}")

    min_e, max_e = compute_min_max(energy_files, pitch_mean, pitch_std)

    print(f"Min energy : {min_e}")
    print(f"Max energy : {max_e}")

