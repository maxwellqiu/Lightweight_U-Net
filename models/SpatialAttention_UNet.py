import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """A spatial attention module.

    This module applies spatial attention to a feature map.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer for spatial attention.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, kernel_size=7):
        """Initializes the SpatialAttention module.

        Args:
            kernel_size (int, optional): The kernel size for the convolutional layer. Defaults to 7.
        """
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of the spatial attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with spatial attention applied.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_pool = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_pool)
        return self.sigmoid(x_att)


class SpatialAttentionUNet(nn.Module):
    """A U-Net model with spatial attention for image segmentation.

    This class implements a U-Net architecture that incorporates spatial attention
    modules in the decoder path.

    Attributes:
        encoder1 (nn.Sequential): First encoding block.
        pool1 (nn.MaxPool2d): First max pooling layer.
        encoder2 (nn.Sequential): Second encoding block.
        pool2 (nn.MaxPool2d): Second max pooling layer.
        encoder3 (nn.Sequential): Third encoding block.
        pool3 (nn.MaxPool2d): Third max pooling layer.
        encoder4 (nn.Sequential): Fourth encoding block.
        pool4 (nn.MaxPool2d): Fourth max pooling layer.
        bottleneck (nn.Sequential): Bottleneck convolutional block.
        upconv4 (nn.ConvTranspose2d): First up-convolution layer.
        sa4 (SpatialAttention): First spatial attention module.
        decoder4 (nn.Sequential): First decoding block.
        upconv3 (nn.ConvTranspose2d): Second up-convolution layer.
        sa3 (SpatialAttention): Second spatial attention module.
        decoder3 (nn.Sequential): Second decoding block.
        upconv2 (nn.ConvTranspose2d): Third up-convolution layer.
        sa2 (SpatialAttention): Third spatial attention module.
        decoder2 (nn.Sequential): Third decoding block.
        upconv1 (nn.ConvTranspose2d): Fourth up-convolution layer.
        sa1 (SpatialAttention): Fourth spatial attention module.
        decoder1 (nn.Sequential): Fourth decoding block.
        final_conv (nn.Conv2d): Final convolutional layer to produce the output.
    """

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        """Initializes the SpatialAttentionUNet model.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            init_features (int, optional): Number of initial features. Defaults to 32.
        """
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
        """Forward pass of the SpatialAttentionUNet model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output segmentation map.
        """
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
    """A U-Net model with spatial attention, modified for Barlow Twins.

    This class adapts the SpatialAttentionUNet for self-supervised learning
    with Barlow Twins by separating the encoder and decoder forward passes.

    Attributes:
        encoder1 (nn.Sequential): First encoding block.
        pool1 (nn.MaxPool2d): First max pooling layer.
        encoder2 (nn.Sequential): Second encoding block.
        pool2 (nn.MaxPool2d): Second max pooling layer.
        encoder3 (nn.Sequential): Third encoding block.
        pool3 (nn.MaxPool2d): Third max pooling layer.
        encoder4 (nn.Sequential): Fourth encoding block.
        pool4 (nn.MaxPool2d): Fourth max pooling layer.
        bottleneck (nn.Sequential): Bottleneck convolutional block.
        upconv4 (nn.ConvTranspose2d): First up-convolution layer.
        sa4 (SpatialAttention): First spatial attention module.
        decoder4 (nn.Sequential): First decoding block.
        upconv3 (nn.ConvTranspose2d): Second up-convolution layer.
        sa3 (SpatialAttention): Second spatial attention module.
        decoder3 (nn.Sequential): Second decoding block.
        upconv2 (nn.ConvTranspose2d): Third up-convolution layer.
        sa2 (SpatialAttention): Third spatial attention module.
        decoder2 (nn.Sequential): Third decoding block.
        upconv1 (nn.ConvTranspose2d): Fourth up-convolution layer.
        sa1 (SpatialAttention): Fourth spatial attention module.
        decoder1 (nn.Sequential): Fourth decoding block.
        final_conv (nn.Conv2d): Final convolutional layer to produce the output.
    """

    def __init__(self, init_features, in_channels=1, out_channels=1):
        """Initializes the SpatialAttentionUNet_Barlow model.

        Args:
            init_features (int): Number of initial features.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
        """
        super(SpatialAttentionUNet_Barlow, self).__init__()
        features = init_features

        def _conv_block(in_c, out_c):
            """A helper function to create a convolutional block.

            Args:
                in_c (int): Input channels.
                out_c (int): Output channels.

            Returns:
                nn.Sequential: A sequential container of layers.
            """
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
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]: A tuple containing the bottleneck tensor
                and a tuple of skip connection tensors.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        return bottleneck, (enc1, enc2, enc3, enc4)

    def forward_decoder(self, bottleneck, skips):
        """Forward pass through the decoder.

        Args:
            bottleneck (torch.Tensor): The bottleneck tensor from the encoder.
            skips (Tuple[torch.Tensor, ...]): A tuple of skip connection tensors.

        Returns:
            torch.Tensor: The output segmentation map.
        """
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
        """Forward pass of the SpatialAttentionUNet_Barlow model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output segmentation map.
        """
        bottleneck, skips = self.forward_encoder(x)
        output = self.forward_decoder(bottleneck, skips)
        return output
