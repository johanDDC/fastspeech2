import torch
from torch import nn

from src.model.decoder.decoder import Decoder
from src.model.encoder.encoder import Encoder
from src.model.utils.utils import get_mask_from_lengths
from src.model.var_adaptor.var_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size,
                 max_len,
                 n_layers,
                 d_model,
                 d_inner,
                 pad_idx,
                 conv_block_kernel_size,
                 variance_kernel_size,
                 n_heads,
                 num_mels,
                 variance_dropout_prob,
                 conv_dropout_prob,
                 pitch_energy_stats,
                 device):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(n_layers, max_len, vocab_size, d_model, d_inner,
                               conv_block_kernel_size, n_heads, conv_dropout_prob, pad_idx)
        self.decoder = Decoder(n_layers, max_len, vocab_size, d_model, d_inner,
                               conv_block_kernel_size, n_heads, conv_dropout_prob, pad_idx)
        self.var_adaptor = VarianceAdaptor(d_model, variance_kernel_size, variance_dropout_prob,
                                           pitch_energy_stats, device)
        self.mel_linear = nn.Linear(d_model, num_mels)
        self.device = device

    def forward(self, src_seq, src_pos, input_lengths, max_input_length,
                mel_pos=None, output_lengths=None, max_output_length=None,
                duration_target=None, energy_target=None, pitch_target=None,
                duration_alpha=1, pitch_alpha=1, energy_alpha=1):
        src_masks = get_mask_from_lengths(input_lengths, max_input_length)
        mel_masks = get_mask_from_lengths(output_lengths, max_output_length) \
            if output_lengths is not None else None

        out = self.encoder(src_seq, src_pos, src_masks)

        (out, log_duration_prediction, pitch_prediction, energy_prediction), \
        (round_duration, decoder_mel_masks) = self.var_adaptor(out, src_masks, mel_masks,
                                                      max_output_length,
                                                      duration_target,
                                                      pitch_target,
                                                      energy_target,
                                                      duration_alpha,
                                                      pitch_alpha, energy_alpha)

        mel_pos = mel_pos if mel_pos is not None else torch.stack(
            [torch.arange(1, out.shape[1] + 1, dtype=torch.long, device=self.device)]
        )
        out = self.decoder(out, mel_pos, decoder_mel_masks)
        out = self.mel_linear(out)

        return (out, log_duration_prediction, pitch_prediction, energy_prediction), \
               (round_duration, src_masks, decoder_mel_masks)
