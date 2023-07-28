import torch
import torch.nn as nn


class Spring(nn.Module):
    """
    Custom made Spring module, modified to suit the inception version of the architecture
    It utilizes channel-wise pooling, allowing us to build a deeper network without exploding the number of channels
    Number of output channels is equal to number of input channels.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1, skip=False, kernel=3):
        super().__init__()
        self.skip = skip
        if not mid_channels:
            mid_channels = out_channels
        self.spring = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel, kernel), padding=kernel//2, stride=(1, 1), bias=False, dtype=torch.float32),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=False, dtype=torch.float32),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x, skip_data=None):
        if self.skip:
            return skip_data + self.spring(x)
        else:
            return self.spring(x)


class MemoryBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.1, skip=False, kernel=3):
        super().__init__()
        self.skip = skip
        self.spring = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=False, dtype=torch.float32),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x, skip_data=None):
        if self.skip:
            return skip_data + self.spring(x)
        else:
            return self.spring(x)