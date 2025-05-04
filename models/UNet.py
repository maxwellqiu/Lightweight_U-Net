import torch
import torch.nn as nn


class UNetBaseline(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, init_feat=32):
        super().__init__()

        def conv_block(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                                 nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

        f = init_feat
        self.enc1 = conv_block(in_ch, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.dec4 = conv_block(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.dec3 = conv_block(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.dec2 = conv_block(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.dec1 = conv_block(f * 2, f)
        self.final = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)
