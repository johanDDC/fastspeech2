import torch
from torch.utils.data import Dataset, DataLoader
from .texts import phonemes_to_sequence
import numpy as np

from utils import pad_1D_tensor, pad_2D_tensor


def make_dataset(path, batch_size, cfg, valid=False):
    metafile = cfg.data.valid_meta if valid else cfg.data.train_meta
    stats = (cfg.data.pitch_mean, cfg.data.pitch_std,
             cfg.data.energy_mean, cfg.data.energy_std)
    dataset = TTSDataset(path, metafile, cfg.data.text_cleaners, stats=stats)

    train_set = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn_tensor,
        shuffle=(not valid),
        pin_memory=(not valid),
    )
    return train_set


class TTSDataset(Dataset):
    def __init__(self, path, file_, tts_cleaner_names, stats):
        self.path = path
        self.p_mean, self.p_std, self.e_mean, self.e_std = stats
        with open(file_, encoding="utf-8") as f:
            self._metadata = [line.strip().split("|") for line in f]
        self.tts_cleaner_names = tts_cleaner_names

    def __getitem__(self, index):
        file_name = self._metadata[index][4].split(".")[0]
        phonemes_seq = self._metadata[index][3].split()
        x = phonemes_to_sequence(phonemes_seq)
        mel = np.load(f"{self.path}/mels/{file_name}.npy")
        durations = list(map(int, self._metadata[index][2].split()))
        e = np.load(f"{self.path}/energy/{file_name}.npy")
        e = (e - self.e_mean) / self.e_std
        p = np.load(f"{self.path}/pitch/{file_name}.npy")
        p = (p - self.p_mean) / self.p_std
        return {
            "phonemes": np.array(x),
            "mel_target": mel.T,
            "duration": np.array(durations),
            "energy": e,
            "pitch": p,
        }

    def __len__(self):
        return len(self._metadata)


def reprocess_tensor(batch, idx):
    texts = [torch.from_numpy(batch[ind]["phonemes"]) for ind in idx]
    mel_targets = [torch.from_numpy(batch[ind]["mel_target"]) for ind in idx]
    durations = [torch.from_numpy(batch[ind]["duration"]) for ind in idx]
    energy = [torch.from_numpy(batch[ind]["energy"]) for ind in idx]
    pitch = [torch.from_numpy(batch[ind]["pitch"]) for ind in idx]

    length_text = np.array([text.shape[0] for text in texts])
    length_mel = np.array([mel.shape[0] for mel in mel_targets])

    src_pos = list()
    max_len = max(length_text)
    for length_src_row in length_text:
        src_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                              (0, max_len - int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    mel_pos = []
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                              (0, max_mel_len - int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitches = pad_1D_tensor(pitch)
    energies = pad_1D_tensor(energy)

    out = {
        "src": texts,
        "src_length": torch.from_numpy(length_text),
        "max_len": max_len,
        "mel_target": mel_targets,
        "mel_length": torch.from_numpy(length_mel),
        "duration": durations,
        "mel_pos": mel_pos,
        "src_pos": src_pos,
        "mel_max_len": max_mel_len,
        "pitch": pitches,
        "energy": energies,
    }

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["phonemes"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)

    return reprocess_tensor(batch, index_arr)
