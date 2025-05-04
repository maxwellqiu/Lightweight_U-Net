import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_pool = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_pool)
        return self.sigmoid(x_att)


class SpatialAttentionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(SpatialAttentionUNet, self).__init__()
        features = init_features

        _conv_block = lambda in_c, out_c: nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = _conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _conv_block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16,
                                          features * 8,
                                          kernel_size=2,
                                          stride=2)
        self.sa4 = SpatialAttention()
        self.decoder4 = _conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8,
                                          features * 4,
                                          kernel_size=2,
                                          stride=2)
        self.sa3 = SpatialAttention()
        self.decoder3 = _conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4,
                                          features * 2,
                                          kernel_size=2,
                                          stride=2)
        self.sa2 = SpatialAttention()
        self.decoder2 = _conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2,
                                          features,
                                          kernel_size=2,
                                          stride=2)
        self.sa1 = SpatialAttention()
        self.decoder1 = _conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)

        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4,
                                 size=enc4.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
        cat4 = torch.cat((dec4, enc4), dim=1)
        att4 = self.sa4(cat4)
        dec4_out = self.decoder4(cat4 * att4)

        dec3 = self.upconv3(dec4_out)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3,
                                 size=enc3.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
        cat3 = torch.cat((dec3, enc3), dim=1)
        att3 = self.sa3(cat3)
        dec3_out = self.decoder3(cat3 * att3)

        dec2 = self.upconv2(dec3_out)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2,
                                 size=enc2.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
        cat2 = torch.cat((dec2, enc2), dim=1)
        att2 = self.sa2(cat2)
        dec2_out = self.decoder2(cat2 * att2)

        dec1 = self.upconv1(dec2_out)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1,
                                 size=enc1.shape[2:],
                                 mode='bilinear',
                                 align_corners=False)
        cat1 = torch.cat((dec1, enc1), dim=1)
        att1 = self.sa1(cat1)
        dec1_out = self.decoder1(cat1 * att1)

        return self.final_conv(dec1_out)


class SpatialAttentionUNet_Barlow(nn.Module):

    def __init__(self, init_features, in_channels=1, out_channels=1):
        super(SpatialAttentionUNet_Barlow, self).__init__()
        features = init_features

        def _conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = _conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _conv_block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16,
                                          features * 8,
                                          kernel_size=2,
                                          stride=2)
        self.sa4 = SpatialAttention()
        self.decoder4 = _conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8,
                                          features * 4,
                                          kernel_size=2,
                                          stride=2)
        self.sa3 = SpatialAttention()
        self.decoder3 = _conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4,
                                          features * 2,
                                          kernel_size=2,
                                          stride=2)
        self.sa2 = SpatialAttention()
        self.decoder2 = _conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2,
                                          features,
                                          kernel_size=2,
                                          stride=2)
        self.sa1 = SpatialAttention()
        self.decoder1 = _conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        return bottleneck, (enc1, enc2, enc3, enc4)

    def forward_decoder(self, bottleneck, skips):
        enc1, enc2, enc3, enc4 = skips
        dec4 = self.upconv4(bottleneck)
        cat4 = torch.cat((dec4, enc4), dim=1)
        att4 = self.sa4(cat4)
        dec4_out = self.decoder4(cat4 * att4)

        dec3 = self.upconv3(dec4_out)
        cat3 = torch.cat((dec3, enc3), dim=1)
        att3 = self.sa3(cat3)
        dec3_out = self.decoder3(cat3 * att3)

        dec2 = self.upconv2(dec3_out)
        cat2 = torch.cat((dec2, enc2), dim=1)
        att2 = self.sa2(cat2)
        dec2_out = self.decoder2(cat2 * att2)

        dec1 = self.upconv1(dec2_out)
        cat1 = torch.cat((dec1, enc1), dim=1)
        att1 = self.sa1(cat1)
        dec1_out = self.decoder1(cat1 * att1)
        return self.final_conv(dec1_out)

    def forward(self, x):
        bottleneck, skips = self.forward_encoder(x)
        output = self.forward_decoder(bottleneck, skips)
        return output
