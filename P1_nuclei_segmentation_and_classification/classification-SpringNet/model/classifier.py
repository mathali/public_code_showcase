""" SpringNet implementation """

import torch
from torch import nn

from .classifier_parts import *


class Classifier(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Classifier, self).__init__()
        torch.set_default_dtype(torch.float32)

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout_small = nn.Dropout(0.2)
        self.dropout_medium = nn.Dropout(0.3)
        self.dropout_large = nn.Dropout(0.4)

        self.act = nn.LeakyReLU()

        self.input_conv = nn.Conv2d(self.n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.input_bn = nn.BatchNorm2d(64)

        self.small_spring_1 = Spring(in_channels=64, out_channels=64, mid_channels=128, dropout=0.1)
        self.small_spring_2 = Spring(in_channels=64, out_channels=64, mid_channels=128, dropout=0.1, skip=True)

        self.up_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.up_1_bn = nn.BatchNorm2d(128)

        self.medium_spring_1 = Spring(in_channels=128, out_channels=128, mid_channels=256, dropout=0.2)
        self.medium_spring_2 = Spring(in_channels=128, out_channels=128, mid_channels=256, dropout=0.2, skip=True)

        self.up_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.up_2_bn = nn.BatchNorm2d(256)

        self.large_spring_1 = Spring(in_channels=256, out_channels=256, mid_channels=512, dropout=0.3)
        self.large_spring_2 = Spring(in_channels=256, out_channels=256, mid_channels=512, dropout=0.3, skip=True)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.n_classes)

    def forward(self, x):
        x1 = self.act(self.input_bn(self.input_conv(x)))
        x1 = self.dropout_small(x1)

        x2 = self.small_spring_1(x1)
        x2 = self.small_spring_2(x2, skip_data=x1)

        x2 = self.act(self.up_1_bn(self.up_1(x2)))
        x2 = self.dropout_small(x2)

        x3 = self.medium_spring_1(x2)
        x3 = self.medium_spring_2(x3, skip_data=x2)

        x3 = self.act(self.up_2_bn(self.up_2(x3)))
        x3 = self.dropout_medium(x3)

        x4 = self.large_spring_1(x3)
        x4 = self.large_spring_2(x4, skip_data=x3)

        x4 = torch.flatten(x4, 1)

        x4 = self.act(self.fc1(x4))
        x4 = self.dropout_large(x4)

        x4 = self.act(self.fc2(x4))
        x4 = self.dropout_large(x4)

        x4 = self.fc3(x4)

        return x4

