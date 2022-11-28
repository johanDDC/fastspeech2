import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, inputs, predictions, masks):
        src_mask, mel_mask = masks
        src_mask = ~src_mask
        mel_mask = ~mel_mask

        mels_target, duration_target, pitch_target, energy_target = inputs
        predicted_mels, log_duration_prediction, pitch_prediction, energy_prediction = predictions

        predicted_mels = predicted_mels.masked_select(mel_mask.unsqueeze(-1))
        mels_target = mels_target.masked_select(mel_mask.unsqueeze(-1))
        mel_loss = F.l1_loss(predicted_mels, mels_target)

        log_duration_target = torch.log(duration_target + 1)
        log_duration_prediction = log_duration_prediction.masked_select(src_mask)
        log_duration_target = log_duration_target.masked_select(src_mask)
        duration_loss = F.mse_loss(log_duration_prediction, log_duration_target)

        pitch_prediction = pitch_prediction.masked_select(mel_mask)
        pitch_target = pitch_target.masked_select(mel_mask)
        pitch_loss = F.mse_loss(pitch_prediction, pitch_target)

        energy_prediction = energy_prediction.masked_select(mel_mask)
        energy_target = energy_target.masked_select(mel_mask)
        energy_loss = F.mse_loss(energy_prediction, energy_target)

        return (
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
