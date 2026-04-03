"""
ResNet18-UNet: Pretrained ResNet18 encoder with U-Net decoder.

Baseline model from:
"Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation"
Qiu et al., 2025

Parameters: 12.69M | GFLOPs: 3.28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18_UNet(nn.Module):
    """
    U-Net with pretrained ResNet18 encoder.

    Skip connections from layer1 (64ch), layer2 (128ch),
    layer3 (256ch), layer4 (512ch).
    """
    def __init__(self, out_channels=1, dropout_p=0.3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1  # 64 ch
        self.enc2 = resnet.layer2  # 128 ch
        self.enc3 = resnet.layer3  # 256 ch
        self.enc4 = resnet.layer4  # 512 ch

        self.bottle = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.dropout = nn.Dropout2d(dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.dec3 = self._dec_block(256 + 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec2 = self._dec_block(128 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = self._dec_block(64 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def _dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        input_size = x.shape[2:]

        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.dropout(self.bottle(e4))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:],
                               mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], 1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:],
                               mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], 1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:],
                               mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], 1))

        out = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        return self.final(out)
