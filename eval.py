import os
import torch
import yaml
import warnings

import waveglow
from src.data.texts import valid_symbols, phonemes_to_sequence
from src.data.texts.cleaners import english_cleaners, punctuation_removers
from src.model.Fastspeech2 import FastSpeech2
from utils import Dict, get_WaveGlow
from g2p_en import G2p

warnings.filterwarnings("ignore")

def text_to_phonemes(sentence, g2p):
    clean_content = english_cleaners(sentence)
    clean_content = punctuation_removers(clean_content)
    phonemes = g2p(clean_content)
    phonemes = ["" if x == " " else x for x in phonemes]
    phonemes = ["pau" if x in [",", "."] else x for x in phonemes]
    phonemes = " ".join(phonemes)
    return phonemes

@torch.no_grad()
def synthesis(model, sentence, device, g2p,
              duration_control=1, pitch_control=1, energy_control=1):
    phonemes = text_to_phonemes(sentence, g2p)
    phonemes = phonemes_to_sequence(phonemes)
    phonemes = torch.tensor(phonemes, device=device)
    input_len = torch.tensor(phonemes.shape, dtype=torch.long, device=device)
    src_pos = torch.arange(1, input_len[0] + 1).to(device)

    (out, _, _, _), _ = model(phonemes.unsqueeze(0), src_pos, input_len, input_len.item(),
                              duration_alpha=duration_control,
                              pitch_alpha=pitch_control,
                              energy_alpha=energy_control)

    return out.transpose(1, 2)

def validation(model, val_config):
    WaveGlow = get_WaveGlow()
    g2p = G2p()
    model.eval()

    with open(val_config.val.val_texts) as f:
        val_texts = f.readlines()

    for i, val_sentence in enumerate(val_texts):
        os.makedirs("results", exist_ok=True)
        duration_control = val_config.val.duration_control
        pitch_control = val_config.val.pitch_control
        energy_control = val_config.val.energy_control
        mel = synthesis(model, val_sentence, val_config.model.base_settings.device, g2p,
                        duration_control, pitch_control, energy_control)
        waveglow.inference.inference(
            mel, WaveGlow,
            f"results/d={duration_control}_p={pitch_control}_e={energy_control}_{i}_waveglow.wav"
        )


if __name__ == '__main__':
    cfg = yaml.load(open("./config/fastspeech2.yaml", "r"), Loader=yaml.FullLoader)
    cfg = Dict(cfg)
    model_config = cfg.model
    data_config = cfg.data
    train_config = cfg.train

    vocab_size = len(valid_symbols)
    num_mels = cfg.data.num_mels
    pitch_energy_stats = (cfg.data.pitch_min, cfg.data.pitch_max,
                          cfg.data.energy_min,cfg.data.energy_max)

    model = FastSpeech2(vocab_size=vocab_size,
                        max_len=model_config.base_settings.max_len,
                        n_layers=model_config.base_settings.n_layers,
                        d_model=model_config.base_settings.hidden_dim,
                        d_inner=model_config.base_settings.conv_filter_size,
                        pad_idx=data_config.pad_idx,
                        conv_block_kernel_size=model_config.base_settings.conv_kernel_size,
                        variance_kernel_size=model_config.variance_predictor.kernel_size,
                        n_heads=model_config.base_settings.attention_heads,
                        num_mels=model_config.base_settings.num_mels,
                        variance_dropout_prob=model_config.variance_predictor.dropout,
                        conv_dropout_prob=model_config.base_settings.dropout,
                        pitch_energy_stats=pitch_energy_stats,
                        device=model_config.base_settings.device)

    if os.path.exists(cfg.val.model_path):
        model.load_state_dict(torch.load(cfg.val.model_path)["model"])
    else:
        print("No pretrained model")
        exit(1)

    model.to(cfg.model.base_settings.device)
    validation(model, cfg)
