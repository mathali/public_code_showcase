
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm='batch', groups=16):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm == 'batch':
            midNorm = nn.BatchNorm2d(mid_channels)
            outNorm = nn.BatchNorm2d(out_channels)
        else:
            midNorm = nn.GroupNorm(groups, mid_channels)
            outNorm = nn.GroupNorm(groups, out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dtype=torch.float32),
            midNorm,
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=torch.float32),
            outNorm,
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch', groups=16):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm='batch', groups=16):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, groups=groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=torch.float32)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUp(nn.Module):
    """
    Standard U-Net Upsample block that's been modified to handle the attention alterations
    - meaning that it can handle a varying amount of channels
    """
    def __init__(self, in_channels, out_channels, up_channels, norm='batch', groups=16):
        super().__init__()

        self.up = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=2, stride=2, dtype=torch.float32)
        self.conv = DoubleConv(in_channels, out_channels, norm=norm, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.float32)

    def forward(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention part of our custom attention block.
    Based on the attention U-Net
    Described in the thesis in section 6.3.2
    """
    def __init__(self, in_channels, out_channels, norm='batch'):
        super(SpatialAttention, self).__init__()
        self.x_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.g_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.att_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        x_l = self.x_conv(x)
        g = self.g_conv(g)

        att = torch.add(x_l, g)
        att = self.relu(att)
        att = self.att_conv(att)

        att = self.sigmoid(att)
        att = self.up(att)

        return torch.matmul(x, att), att


class EncoderAttention(nn.Module):
    """
    Encoder attention part of our custom attention block.
    Extended the idea of channel-wise attention
    Only handles data without the contextual WSI information
    Described in the thesis in section 6.3.2
    """
    def __init__(self, e1_channels, e2_channels, H, W):
        super(EncoderAttention, self).__init__()
        self.pool = torch.nn.AvgPool2d((H, W))
        self.linear = torch.nn.Linear(e1_channels + e2_channels, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, e1, e2):
        e1 = self.pool(e1)
        e2 = self.pool(e2)

        e1 = torch.squeeze(e1)
        e2 = torch.squeeze(e2)

        x = torch.cat([e1, e2], dim=1)
        x = self.linear(x)

        return self.softmax(x)


class PyramidEncoderAttention(nn.Module):
    """
    Encoder attention part of our custom attention block.
    Extended the idea of channel-wise attention
    Final version of the encoder attention module, can handle all input data
    Described in the thesis in section 6.3.2
    """
    def __init__(self, e1_channels, e2_channels, e3_channels, H, W):
        super(PyramidEncoderAttention, self).__init__()
        self.pool = torch.nn.AvgPool2d((H, W))
        self.linear = torch.nn.Linear(e1_channels + e2_channels + e3_channels, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, e1, e2, e3):
        e1 = self.pool(e1)
        e2 = self.pool(e2)
        e3 = self.pool(e3)

        e1 = torch.squeeze(e1)
        e2 = torch.squeeze(e2)
        e3 = torch.squeeze(e3)

        x = torch.cat([e1, e2, e3], dim=1)
        x = self.linear(x)

        return self.softmax(x)