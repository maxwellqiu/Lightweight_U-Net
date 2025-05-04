import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1),
                                nn.ReLU(True),
                                nn.Conv2d(channel // reduction, channel, 1),
                                nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch,
                            in_ch,
                            3,
                            padding=1,
                            groups=in_ch,
                            bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.relu(self.bn(x))


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads=4, layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        b, c, h, w = x.shape
        xf = x.flatten(2).permute(2, 0, 1)
        xf = self.encoder(xf)
        return xf.permute(1, 2, 0).view(b, c, h, w)


class MultiScaleFusion(nn.Module):

    def __init__(self, in_ch, out_ch, scales=[1, 2, 4]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=s,
                            mode='bilinear',
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(True)) for s in scales
        ])
        self.se = SEBlock(out_ch * len(scales))
        self.project = nn.Conv2d(out_ch * len(scales), out_ch, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        mh, mw = max(f.shape[2] for f in feats), max(f.shape[3] for f in feats)
        feats = [
            F.interpolate(f,
                          size=(mh, mw),
                          mode='bilinear',
                          align_corners=False) for f in feats
        ]
        x_cat = torch.cat(feats, dim=1)
        return self.project(self.se(x_cat))


class MobileNetV2_UNet_Attn_MS(nn.Module):

    def __init__(self, dropout_p, img_size, out_channels=1):
        super().__init__()
        self.img_size = img_size
        net = models.mobilenet_v2(pretrained=True)
        self.encoder = net.features
        idx = [1, 3, 6, 13]
        ch = [self.encoder[i].out_channels for i in idx]
        self.bottle = nn.Conv2d(net.features[-1].out_channels, ch[-1], 1)
        self.se = SEBlock(ch[-1])
        self.trans = TransformerBlock(ch[-1])
        self.msf = MultiScaleFusion(ch[-1], ch[-1])
        self.dropout = nn.Dropout2d(dropout_p)
        self.refine = nn.Conv2d(ch[-1], ch[-1], 1)
        self.up_convs, self.dec_convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, 2))
            self.dec_convs.append(
                DepthwiseSeparableConv(ch[i - 1] * 2, ch[i - 1]))
        self.final = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 6, 13]: skips.append(x)
        x = self.bottle(x)
        x = self.se(x)
        x = self.trans(x)
        x = self.msf(x)
        x = self.dropout(x)
        x = F.relu(self.refine(x))
        for up, dec, skip in zip(self.up_convs, self.dec_convs,
                                 reversed(skips[:-1])):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x,
                                  size=skip.shape[2:],
                                  mode='bilinear',
                                  align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        x = F.interpolate(x,
                          size=self.img_size,
                          mode='bilinear',
                          align_corners=False)
        return torch.sigmoid(self.final(x))
