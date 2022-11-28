import torch
from torch import nn
import torch.nn.functional as F


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def _repeat_sequence(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if d.sum() == 0:
            d = d.fill_(1)
        out = []
        for x_, d_ in zip(x, d):
            if d_ != 0:
                out.append(x_.repeat(int(d_), 1))

        return torch.cat(out, dim=0)

    def LR(self, x, duration, max_len):
        out = []
        mel_len = []
        for batch, repeat_duration in zip(x, duration):
            repeated_mel = self.repeat(batch, repeat_duration)
            mel_len.append(repeated_mel.shape[0])
            out.append(repeated_mel)

        if max_len is None:
            max_len = max([out[i].shape[0] for i in range(len(out))])

        for i in range(len(out)):
            out[i] = F.pad(out[i], (0, 0, 0, max_len - out[i].size(0)))

        return torch.stack(out), torch.tensor(mel_len, dtype=torch.long, device=x.device)

    def repeat(self, batch, predicted):
        out = [vec.expand(predicted[i].int().item(), -1) for i, vec in enumerate(batch)]
        return torch.cat(out, 0)

    def forward(self, x, duration, mel_max_len):
        return self.LR(x, duration, mel_max_len)
