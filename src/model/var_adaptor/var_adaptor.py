import torch
from torch import nn

from src.model.utils.utils import get_mask_from_lengths
from src.model.var_adaptor.length_regularizator import LengthRegulator
from src.model.var_adaptor.var_adaptor_block import VarianceAdaptorBlock


class VarianceAdaptor(nn.Module):
    def __init__(self, d_inner, kernel_size, dropout, pitch_energy_stats, device):
        super().__init__()
        self.duration_predictor = VarianceAdaptorBlock(d_inner, kernel_size, dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VarianceAdaptorBlock(d_inner, kernel_size, dropout)
        self.energy_predictor = VarianceAdaptorBlock(d_inner, kernel_size, dropout)

        pitch_min, pitch_max, energy_min, energy_max = pitch_energy_stats

        self.pitch_quant_bins = torch.linspace(pitch_min, pitch_max, 255).to(device)
        self.energy_quant_bins = torch.linspace(energy_min, energy_max, 255).to(device)
        self.pitch_embedding = nn.Embedding(256, d_inner).to(device)
        self.energy_embedding = nn.Embedding(256, d_inner).to(device)

    def forward(self, x, src_mask,
                mel_mask=None, max_len=None,
                duration_target=None, pitch_target=None, energy_target=None,
                duration_alpha=1, pitch_alpha=1, energy_alpha=1):

        log_duration_predicted = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, _ = self.length_regulator(x, duration_target, max_len)
            round_duration = duration_target
        else:
            round_duration = torch.clamp(
                (torch.exp(log_duration_predicted) - 1).round() * duration_alpha,
                min=0)
            x, mel_length = self.length_regulator(x, round_duration, max_len)
            mel_mask = get_mask_from_lengths(mel_length)

        pitch_prediction = self.pitch_predictor(x, mel_mask)
        energy_prediction = self.energy_predictor(x, mel_mask)
        if pitch_target is not None:
            pitch_embeddings = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_quant_bins))
        else:
            pitch_embeddings = self.pitch_embedding(
                torch.bucketize(pitch_prediction * pitch_alpha, self.pitch_quant_bins)
            )
        if energy_target is not None:
            energy_embeddings = self.energy_embedding(torch.bucketize(energy_target, self.energy_quant_bins))
        else:
            energy_embeddings = self.energy_embedding(
                torch.bucketize(energy_prediction * energy_alpha, self.energy_quant_bins)
            )
        x = x + pitch_embeddings + energy_embeddings

        return (x, log_duration_predicted, pitch_prediction, energy_prediction), \
               (round_duration, mel_mask)
