""" See DISCLAIMER.txt """
from .unet_parts import *
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.dropout = nn.Dropout(0.2)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        #########################
        x = self.up1(x5, x4)
        x = self.dropout(x)
        
        x = self.up2(x, x3)
        x = self.dropout(x)
        
        x = self.up3(x, x2)
        x = self.dropout(x)
        
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
