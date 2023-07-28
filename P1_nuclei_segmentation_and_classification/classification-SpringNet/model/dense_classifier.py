""" SpringNet implementation
Uses DenseNet skip connections
Only used during experiments, did not prove to be more successful than the 'classic' SpringNet implementation.
"""

import torch
from torch import nn

from .classifier_parts import *


class DenseClassifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DenseClassifier, self).__init__()
        torch.set_default_dtype(torch.float32)

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout_small = nn.Dropout(0.2)
        self.dropout_medium = nn.Dropout(0.3)
        self.dropout_large = nn.Dropout(0.4)

        self.act = nn.LeakyReLU()

        self.input_conv = nn.Conv2d(self.n_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.input_bn = nn.BatchNorm2d(32)

        self.small_spring_1 = Spring(in_channels=32, out_channels=32, mid_channels=64, dropout=0.35)
        self.small_spring_2 = Spring(in_channels=64, out_channels=64, mid_channels=128, dropout=0.35)

        self.up_1 = nn.Conv2d(160, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.up_1_bn = nn.BatchNorm2d(256)

        self.medium_spring_1 = Spring(in_channels=256, out_channels=256, mid_channels=512, dropout=0.35)
        self.medium_spring_2 = Spring(in_channels=512, out_channels=512, mid_channels=1024, dropout=0.35)

        self.up_2 = nn.Conv2d(1280, 2048, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.up_2_bn = nn.BatchNorm2d(2048)

        self.pool = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, self.n_classes)

    def forward(self, x):
        x1 = self.act(self.input_bn(self.input_conv(x)))
        x1 = self.dropout_small(x1)

        x2_1 = self.small_spring_1(x1)
        x2_2 = torch.cat([x1, x2_1], 1)
        x2_3 = self.small_spring_2(x2_2)
        x2_4 = torch.cat([x1, x2_2, x2_3], 1)

        x3 = self.act(self.up_1_bn(self.up_1(x2_4)))
        x3 = self.dropout_medium(x3)

        x4_1 = self.medium_spring_1(x3)
        x4_2 = torch.cat([x3, x4_1], 1)
        x4_3 = self.medium_spring_2(x4_2)
        x4_4 = torch.cat([x3, x4_2, x4_3], 1)

        x5 = self.act(self.up_2_bn(self.up_2(x4_4)))
        x5 = self.dropout_medium(x5)

        x5 = self.pool(x5)

        x5 = torch.flatten(x5, 1)

        x5 = self.act(self.fc1(x5))
        x5 = self.dropout_large(x5)
        x5 = self.fc2(x5)

        return x5

