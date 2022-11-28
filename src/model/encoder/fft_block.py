import torch
from torch import nn

from src.model.encoder.attention import MultiHeadAttention
from src.model.encoder.conv_block import ConvBlock


class FFTBlock(torch.nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 conv_block_kernel_size,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.conv_block = ConvBlock(d_model, d_inner, conv_block_kernel_size, dropout)

    def forward(self, x, non_pad_mask=None, attention_mask=None):
        x = self.ln1(x)
        attention_output, attention_map = self.slf_attn(
            x, mask=attention_mask)

        x = x + attention_output
        if non_pad_mask is not None:
            x = x.masked_fill(non_pad_mask.unsqueeze(-1), 0)

        x = self.ln2(x)
        conv_output = self.conv_block(x)

        x = x + conv_output
        if non_pad_mask is not None:
            x = x.masked_fill(non_pad_mask.unsqueeze(-1), 0)

        return x, attention_map
