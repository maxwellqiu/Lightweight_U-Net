"""
MobileNetV2-UNet: Lightweight encoder with depthwise-separable decoder.

Proposed model from:
"Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation"
Qiu et al., 2025

Parameters: 2.38M | GFLOPs: 0.46
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


class MobileNetV2_UNet(nn.Module):
    """
    U-Net with MobileNetV2 encoder pretrained on ImageNet.

    Skip connections at layers {1, 3, 6, 13} -> {16, 24, 32, 96} channels.
    Decoder uses depthwise-separable convolutions.
    """
    def __init__(self, out_channels=1, dropout_p=0.3):
        super().__init__()
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = net.features
        idx = [1, 3, 6, 13]
        ch = [self.encoder[i].out_channels for i in idx]

        self.bottle = nn.Conv2d(net.features[-1].out_channels, ch[-1], 1)
        self.dropout = nn.Dropout2d(dropout_p)
        self.refine = nn.Conv2d(ch[-1], ch[-1], 1)

        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(ch[i], ch[i-1], 2, 2))
            self.dec_convs.append(DepthwiseSeparableConv(ch[i-1] * 2, ch[i-1]))

        self.final = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 6, 13]:
                skips.append(x)

        x = self.bottle(x)
        x = self.dropout(x)
        x = F.relu(self.refine(x))

        for up, dec, skip in zip(self.up_convs, self.dec_convs, reversed(skips[:-1])):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return self.final(x)
