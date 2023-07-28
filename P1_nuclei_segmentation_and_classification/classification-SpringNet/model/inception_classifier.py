""" SpringNet implementation
Uses Inception-like modifications
Only used during experiments, did not prove to be more successful than the 'classic' SpringNet implementation.
"""
import torch
from torch import nn

from .inception_classifier_parts import *

class InceptionClassifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(InceptionClassifier, self).__init__()
        torch.set_default_dtype(torch.float32)

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout_small = nn.Dropout(0.2)
        self.dropout_medium = nn.Dropout(0.3)
        self.dropout_large = nn.Dropout(0.4)

        self.act = nn.LeakyReLU()

        self.input_conv = nn.Conv2d(self.n_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.input_bn = nn.BatchNorm2d(32)

        self.small_spring_1 = Spring(in_channels=32, out_channels=32, mid_channels=64, dropout=0.1, kernel=3)
        self.small_spring_2 = Spring(in_channels=32, out_channels=32, mid_channels=64, dropout=0.1, kernel=5)
        self.memory_block_1 = MemoryBlock(in_channels=32, out_channels=32, dropout=0.1)

        self.up_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.up_1_bn = nn.BatchNorm2d(128)

        self.medium_spring_1 = Spring(in_channels=128, out_channels=128, mid_channels=256, dropout=0.1, kernel=3)
        self.medium_spring_2 = Spring(in_channels=128, out_channels=128, mid_channels=256, dropout=0.1, kernel=5)
        self.memory_block_2 = MemoryBlock(in_channels=128, out_channels=128, dropout=0.1)

        self.pool = nn.MaxPool2d(7)

        self.fc1 = nn.Linear(384, self.n_classes)

    def forward(self, x):
        x1 = self.act(self.input_bn(self.input_conv(x)))
        x1 = self.dropout_small(x1)

        x2_1 = self.small_spring_1(x1)
        x2_2 = self.small_spring_2(x1)
        x2_3 = self.memory_block_1(x1)

        x2 = torch.cat([x2_1, x2_2, x2_3], 1)

        x3 = self.act(self.up_1_bn(self.up_1(x2)))
        x3 = self.dropout_small(x3)

        x4_1 = self.medium_spring_1(x3)
        x4_2 = self.medium_spring_2(x3)
        x4_3 = self.memory_block_2(x3)

        x4 = torch.cat([x4_1, x4_2, x4_3], 1)
        x4 = self.pool(x4)

        x4 = torch.flatten(x4, 1)

        x4 = self.fc1(x4)

        return x4

