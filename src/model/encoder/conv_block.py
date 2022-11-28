from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout_prob):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size=kernel_size[0], padding='same'),
            nn.ReLU(),
            nn.Conv1d(d_inner, d_model, kernel_size=kernel_size[1], padding='same'))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        layer_out = x.transpose(1, 2)
        layer_out = self.conv(layer_out)
        layer_out = self.dropout(layer_out.transpose(1, 2))
        return layer_out
