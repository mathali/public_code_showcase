import torch
import cv2
import numpy as np
from .unet_parts import *


""" See DISCLAIMER.txt in P1/segmentation"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, norm='batch'):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.dropout = nn.Dropout(0.3)

        self.inc = DoubleConv(n_channels, 64, norm=norm)
        self.down1 = Down(64, 128, norm=norm)
        self.down2 = Down(128, 256, norm=norm)
        self.down3 = Down(256, 512, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm=norm)
        self.up1 = Up(1024, 512 // factor, bilinear, norm=norm)
        self.up2 = Up(512, 256 // factor, bilinear, norm=norm)
        self.up3 = Up(256, 128 // factor, bilinear, norm=norm)
        self.up4 = Up(128, 64, bilinear, norm=norm)
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
        x = self.dropout(x)
        logits = self.outc(x)
        return logits



class AttentionUNet(nn.Module):
    """
    First version of our custom attention U-Net
    This model only utilizes two encoders - close up of tissue and cell distributions.
    Only used in initial experiments, not described in the thesis
    """
    def __init__(self, n_channels, n_classes, bilinear=False, norm='batch'):
        super(AttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.dropout = nn.Dropout(0.3)
        factor = 2 if bilinear else 1

        self.inc1 = DoubleConv(n_channels, 64, norm=norm)
        self.sp_att1_1 = SpatialAttention(64, 128, norm=norm)

        self.down1_1 = Down(64, 128, norm=norm)
        self.sp_att1_2 = SpatialAttention(128, 256, norm=norm)

        self.down1_2 = Down(128, 256, norm=norm)
        self.sp_att1_3 = SpatialAttention(256, 512, norm=norm)

        self.down1_3 = Down(256, 512 // factor, norm=norm)

        self.inc2 = DoubleConv(n_channels, 32, norm=norm)
        self.sp_att2_1 = SpatialAttention(32, 128, norm=norm)

        self.down2_1 = Down(32, 64, norm=norm)
        self.sp_att2_2 = SpatialAttention(64, 256, norm=norm)

        self.down2_2 = Down(64, 128, norm=norm)
        self.sp_att2_3 = SpatialAttention(128, 512, norm=norm)

        self.enc_att1 = EncoderAttention(64, 32, 256, 256)
        self.enc_att2 = EncoderAttention(128, 64, 128, 128)
        self.enc_att3 = EncoderAttention(256, 128, 64, 64)

        self.up1 = AttentionUp(640, 256 // factor, 512, norm=norm)
        self.up2 = AttentionUp(320, 128 // factor, 256, norm=norm)
        self.up3 = AttentionUp(160, 64, 128, norm=norm)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1_1 = self.inc1(x[:, :3, :, :])

        x1_2 = self.down1_1(x1_1)
        x1_2 = self.dropout(x1_2)

        x1_3 = self.down1_2(x1_2)
        x1_3 = self.dropout(x1_3)

        x1_4 = self.down1_3(x1_3)
        x1_4 = self.dropout(x1_4)

        x2_1 = self.inc2(x[:, 3:, :, :])

        x2_2 = self.down2_1(x2_1)
        x2_2 = self.dropout(x2_2)

        x2_3 = self.down2_2(x2_2)
        x2_3 = self.dropout(x2_3)

        #########################
        enc_att_3 = self.enc_att3(x1_3, x2_3)

        x1_3, att1_3 = self.sp_att1_3(x1_3, x1_4)
        x2_3, att2_3 = self.sp_att2_3(x2_3, x1_4)

        x1_3 = torch.mul(x1_3.view(x1_3.shape[0], -1), enc_att_3[:, 0].view(x1_3.shape[0], -1)).view(x1_3.shape)
        x2_3 = torch.mul(x2_3.view(x2_3.shape[0], -1), enc_att_3[:, 1].view(x2_3.shape[0], -1)).view(x2_3.shape)

        ##########
        x = self.up1(x1_4, torch.cat([x1_3, x2_3], dim=1))
        x = self.dropout(x)

        enc_att_2 = self.enc_att2(x1_2, x2_2)

        x1_2, att1_2 = self.sp_att1_2(x1_2, x)
        x2_2, att2_2 = self.sp_att2_2(x2_2, x)

        x1_2 = torch.mul(x1_2.view(x1_2.shape[0], -1), enc_att_2[:, 0].view(x1_2.shape[0], -1)).view(x1_2.shape)
        x2_2 = torch.mul(x2_2.view(x2_2.shape[0], -1), enc_att_2[:, 1].view(x2_2.shape[0], -1)).view(x2_2.shape)

        ##########
        x = self.up2(x, torch.cat([x1_2, x2_2], dim=1))
        x = self.dropout(x)

        enc_att_1 = self.enc_att1(x1_1, x2_1)

        x1_1, att1_1 = self.sp_att1_1(x1_1, x)
        x2_1, att2_1 = self.sp_att2_1(x2_1, x)

        x1_1 = torch.mul(x1_1.view(x1_1.shape[0], -1), enc_att_1[:, 0].view(x1_1.shape[0], -1)).view(x1_1.shape)
        x2_1 = torch.mul(x2_1.view(x2_1.shape[0], -1), enc_att_1[:, 1].view(x2_1.shape[0], -1)).view(x2_1.shape)

        ##########
        x = self.up3(x, torch.cat([x1_1, x2_1], dim=1))

        logits = self.outc(x)
        return logits


class PyramidAttentionUNet(nn.Module):
    """
    Custom three encoder attention U-Net

    First encoder - WSI close up
    Second encoder - Cell distributions
    Third encoder - WSI context

    """
    def __init__(self, n_channels, n_classes, bilinear=False, norm='batch'):
        super(PyramidAttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.dropout = nn.Dropout(0.3)
        factor = 2 if bilinear else 1

        ###  Input and first downsample for each encoder
        self.inc1 = DoubleConv(n_channels, 64, norm=norm)
        self.sp_att1_1 = SpatialAttention(64, 128, norm=norm)

        self.down1_1 = Down(64, 128, norm=norm)
        self.sp_att1_2 = SpatialAttention(128, 256, norm=norm)

        self.down1_2 = Down(128, 256, norm=norm)
        self.sp_att1_3 = SpatialAttention(256, 512, norm=norm)

        self.down1_3 = Down(256, 512 // factor, norm=norm)

        ###  Second downsample for each encoder
        self.inc2 = DoubleConv(n_channels, 32, norm=norm, groups=8)
        self.sp_att2_1 = SpatialAttention(32, 128, norm=norm)

        self.down2_1 = Down(32, 64, norm=norm, groups=8)
        self.sp_att2_2 = SpatialAttention(64, 256, norm=norm)

        self.down2_2 = Down(64, 128, norm=norm, groups=8)
        self.sp_att2_3 = SpatialAttention(128, 512, norm=norm)

        ### Third downsample for each encoder
        self.inc3 = DoubleConv(n_channels, 32, norm=norm, groups=8)
        self.sp_att3_1 = SpatialAttention(32, 128, norm=norm)

        self.down3_1 = Down(32, 64, norm=norm, groups=8)
        self.sp_att3_2 = SpatialAttention(64, 256, norm=norm)

        self.down3_2 = Down(64, 128, norm=norm, groups=8)
        self.sp_att3_3 = SpatialAttention(128, 512, norm=norm)

        ### Encoder attention layers for each step
        self.enc_att1 = PyramidEncoderAttention(64, 32, 32, 256, 256)
        self.enc_att2 = PyramidEncoderAttention(128, 64, 64, 128, 128)
        self.enc_att3 = PyramidEncoderAttention(256, 128, 128, 64, 64)

        ### Entire decoder branch
        self.up1 = AttentionUp(768, 256 // factor, 512, norm=norm)
        self.up2 = AttentionUp(384, 128 // factor, 256, norm=norm)
        self.up3 = AttentionUp(192, 64, 128, norm=norm)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, att_log=False):
        att_dict = {}

        ###  Input and first downsample for each encoder
        x1_1 = self.inc1(x[:, :3, :, :])

        x1_2 = self.down1_1(x1_1)
        x1_2 = self.dropout(x1_2)

        x1_3 = self.down1_2(x1_2)
        x1_3 = self.dropout(x1_3)

        x1_4 = self.down1_3(x1_3)
        x1_4 = self.dropout(x1_4)

        ###  Second downsample for each encoder
        x2_1 = self.inc2(x[:, 3:6, :, :])

        x2_2 = self.down2_1(x2_1)
        x2_2 = self.dropout(x2_2)

        x2_3 = self.down2_2(x2_2)
        x2_3 = self.dropout(x2_3)

        ###  Third downsample for each encoder
        x3_1 = self.inc3(x[:, 6:, :, :])

        x3_2 = self.down3_1(x3_1)
        x3_2 = self.dropout(x3_2)

        x3_3 = self.down3_2(x3_2)
        x3_3 = self.dropout(x3_3)

        #########################
        ## First merge in the encoder
        # Calculate the unified encoder attention
        enc_att_3 = self.enc_att3(x1_3, x2_3, x3_3)

        # Calculate spatial attention for each encoder
        x1_3, att1_3 = self.sp_att1_3(x1_3, x1_4)
        x2_3, att2_3 = self.sp_att2_3(x2_3, x1_4)
        x3_3, att3_3 = self.sp_att3_3(x3_3, x1_4)

        if att_log:
            tmp = np.zeros(shape=(x.shape[0], 3, 256, 256), dtype=np.float32)
            tmp[:, 0, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_3.cpu().numpy()[:, 0], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 1, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_3.cpu().numpy()[:, 1], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 2, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_3.cpu().numpy()[:, 2], axis=(1, 2)), 256, axis=1), 256, axis=2)
            att_dict['enc_att_3'] = tmp
            att_dict['att1_3'] = cv2.resize(att1_3.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att2_3'] = cv2.resize(att2_3.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att3_3'] = cv2.resize(att3_3.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)

        # Merge encoder and spatial attentions
        x1_3 = torch.mul(x1_3.view(x1_3.shape[0], -1), enc_att_3[:, 0].view(x1_3.shape[0], -1)).view(x1_3.shape)
        x2_3 = torch.mul(x2_3.view(x2_3.shape[0], -1), enc_att_3[:, 1].view(x2_3.shape[0], -1)).view(x2_3.shape)
        x3_3 = torch.mul(x3_3.view(x3_3.shape[0], -1), enc_att_3[:, 2].view(x3_3.shape[0], -1)).view(x3_3.shape)

        ##########
        ## Second merge in the encoder
        # Calculate the unified encoder attention
        x = self.up1(x1_4, torch.cat([x1_3, x2_3, x3_3], dim=1))
        x = self.dropout(x)

        enc_att_2 = self.enc_att2(x1_2, x2_2, x3_2)

        # Calculate spatial attention for each encoder
        x1_2, att1_2 = self.sp_att1_2(x1_2, x)
        x2_2, att2_2 = self.sp_att2_2(x2_2, x)
        x3_2, att3_2 = self.sp_att3_2(x3_2, x)

        if att_log:
            tmp = np.zeros(shape=(x.shape[0], 3, 256, 256), dtype=np.float32)
            tmp[:, 0, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_2.cpu().numpy()[:, 0], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 1, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_2.cpu().numpy()[:, 1], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 2, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_2.cpu().numpy()[:, 2], axis=(1, 2)), 256, axis=1), 256, axis=2)
            att_dict['enc_att_2'] = tmp
            att_dict['att1_2'] = cv2.resize(att1_2.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att2_2'] = cv2.resize(att2_2.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att3_2'] = cv2.resize(att3_2.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)

        # Merge encoder and spatial attentions
        x1_2 = torch.mul(x1_2.view(x1_2.shape[0], -1), enc_att_2[:, 0].view(x1_2.shape[0], -1)).view(x1_2.shape)
        x2_2 = torch.mul(x2_2.view(x2_2.shape[0], -1), enc_att_2[:, 1].view(x2_2.shape[0], -1)).view(x2_2.shape)
        x3_2 = torch.mul(x3_2.view(x3_2.shape[0], -1), enc_att_2[:, 2].view(x3_2.shape[0], -1)).view(x3_2.shape)

        ##########
        ## Third merge in the encoder
        # Calculate the unified encoder attention
        x = self.up2(x, torch.cat([x1_2, x2_2, x3_2], dim=1))
        x = self.dropout(x)

        enc_att_1 = self.enc_att1(x1_1, x2_1, x3_1)

        # Calculate spatial attention for each encoder
        x1_1, att1_1 = self.sp_att1_1(x1_1, x)
        x2_1, att2_1 = self.sp_att2_1(x2_1, x)
        x3_1, att3_1 = self.sp_att3_1(x3_1, x)

        if att_log:
            tmp = np.zeros(shape=(x.shape[0], 3, 256, 256), dtype=np.float32)
            tmp[:, 0, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_1.cpu().numpy()[:, 0], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 1, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_1.cpu().numpy()[:, 1], axis=(1, 2)), 256, axis=1), 256, axis=2)
            tmp[:, 2, :, :] = np.repeat(np.repeat(np.expand_dims(enc_att_1.cpu().numpy()[:, 2], axis=(1, 2)), 256, axis=1), 256, axis=2)
            att_dict['enc_att_1'] = tmp
            att_dict['att1_1'] = cv2.resize(att1_1.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att2_1'] = cv2.resize(att2_1.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)
            att_dict['att3_1'] = cv2.resize(att3_1.cpu().numpy().squeeze().transpose(1, 2, 0), (256, 256)).transpose(2, 0, 1)

        # Merge encoder and spatial attentions
        x1_1 = torch.mul(x1_1.view(x1_1.shape[0], -1), enc_att_1[:, 0].view(x1_1.shape[0], -1)).view(x1_1.shape)
        x2_1 = torch.mul(x2_1.view(x2_1.shape[0], -1), enc_att_1[:, 1].view(x2_1.shape[0], -1)).view(x2_1.shape)
        x3_1 = torch.mul(x3_1.view(x3_1.shape[0], -1), enc_att_1[:, 2].view(x3_1.shape[0], -1)).view(x3_1.shape)

        ##########
        # Final upsample
        x = self.up3(x, torch.cat([x1_1, x2_1, x3_1], dim=1))

        logits = self.outc(x)

        if att_log:
            return logits, att_dict
        else:
            return logits