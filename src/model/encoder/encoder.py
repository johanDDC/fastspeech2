from torch import nn

from src.model.encoder.fft_block import FFTBlock


class Encoder(nn.Module):
    def __init__(self, n_layers, max_seq_len, vocab_size, d_model, d_inner,
                 conv_block_kernel_size, n_heads, dropout, PAD=None):
        super().__init__()
        self.n_layers = n_layers
        len_max_seq = max_seq_len
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position_enc = nn.Embedding(n_position, d_model, padding_idx=PAD)
        self.pad = PAD

        self.fft_layers = nn.ModuleList(
            [FFTBlock(d_model, d_inner, n_heads,
                      d_model // n_heads, d_model // n_heads,
                      conv_block_kernel_size, dropout) for _ in range(n_layers)]
        )

    def forward(self, src_seq, src_pos, src_mask=None):
        slf_attn_mask = src_mask.unsqueeze(1).expand(-1, src_seq.shape[1], -1)

        x = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for i in range(self.n_layers):
            x, _ = self.fft_layers[i](x, src_mask, slf_attn_mask)

        return x
