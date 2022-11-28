from torch import nn

from src.model.encoder.fft_block import FFTBlock


class Decoder(nn.Module):
    def __init__(self, n_layers, max_seq_len, vocab_size, d_model, d_inner,
                 conv_block_kernel_size, n_heads, dropout, PAD=None):

        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.pad = PAD
        len_max_seq = max_seq_len
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding(n_position, d_model, padding_idx=PAD)

        self.fft_layers = nn.ModuleList(
            [FFTBlock(d_model, d_inner, n_heads,
                      d_model // n_heads, d_model // n_heads,
                      conv_block_kernel_size, dropout) for _ in range(n_layers)]
        )

    def forward(self, enc_seq, enc_pos, src_mask=None):
        slf_attn_mask = src_mask.unsqueeze(1).expand(-1, enc_seq.shape[1], -1)
        x = enc_seq + self.position_enc(enc_pos)

        for i in range(self.n_layers):
            x, _ = self.fft_layers[i](x, src_mask, slf_attn_mask)

        return x