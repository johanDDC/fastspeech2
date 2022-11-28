from torch import nn


class TransposedConv(nn.Module):
    def __init__(self, d_hidden, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(d_hidden, d_hidden, kernel_size=kernel_size, padding='same')

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1)
        return y


class VarianceAdaptorBlock(nn.Module):
    def __init__(self, d_hidden, kernel_size, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(d_hidden),
            TransposedConv(d_hidden, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
            TransposedConv(d_hidden, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, x, mask=None):
        output = self.layer(x)
        output = self.fc(output).squeeze()
        if mask is not None:
            output = output.masked_fill(mask, 0)
        return output
