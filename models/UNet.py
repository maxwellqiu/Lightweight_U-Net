"""
Base U-Net and Attention U-Net baselines.

Baseline models from:
"Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation"
Qiu et al., 2025

Base U-Net:      Parameters: 7.76M | GFLOPs: 12.11
Attention U-Net: Parameters: 7.76M | GFLOPs: 12.12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseUNet(nn.Module):
    """Standard U-Net with symmetric encoder-decoder and skip connections."""

    def __init__(self, in_ch=3, out_ch=1, init_features=32):
        super().__init__()
        f = init_features

        self.enc1 = self._block(in_ch, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._block(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.dec4 = self._block(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.dec3 = self._block(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.dec2 = self._block(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.dec1 = self._block(f * 2, f)

        self.final = nn.Conv2d(f, out_ch, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class AttentionGate(nn.Module):
    """Spatial attention gate for skip connections."""

    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.Wx = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class AttentionUNet(nn.Module):
    """U-Net with spatial attention gates in skip connections."""

    def __init__(self, in_ch=3, out_ch=1, init_features=32):
        super().__init__()
        f = init_features

        self.enc1 = self._block(in_ch, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._block(f * 8, f * 16)

        self.att4 = AttentionGate(f * 16, f * 8, f * 4)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.dec4 = self._block(f * 16, f * 8)

        self.att3 = AttentionGate(f * 8, f * 4, f * 2)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.dec3 = self._block(f * 8, f * 4)

        self.att2 = AttentionGate(f * 4, f * 2, f)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.dec2 = self._block(f * 4, f * 2)

        self.att1 = AttentionGate(f * 2, f, f // 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.dec1 = self._block(f * 2, f)

        self.final = nn.Conv2d(f, out_ch, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        a4 = self.att4(b, e4)
        d4 = self.dec4(torch.cat([self.up4(b), a4], 1))

        a3 = self.att3(d4, e3)
        d3 = self.dec3(torch.cat([self.up3(d4), a3], 1))

        a2 = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([self.up2(d3), a2], 1))

        a1 = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([self.up1(d2), a1], 1))

        return self.final(d1)
