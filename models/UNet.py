import torch
import torch.nn as nn


class UNetBaseline(nn.Module):
    """A baseline U-Net model for image segmentation.

    This class implements a standard U-Net architecture with a series of
    convolutional blocks for encoding and decoding.

    Attributes:
        enc1 (nn.Sequential): First encoding block.
        pool1 (nn.MaxPool2d): First max pooling layer.
        enc2 (nn.Sequential): Second encoding block.
        pool2 (nn.MaxPool2d): Second max pooling layer.
        enc3 (nn.Sequential): Third encoding block.
        pool3 (nn.MaxPool2d): Third max pooling layer.
        enc4 (nn.Sequential): Fourth encoding block.
        pool4 (nn.MaxPool2d): Fourth max pooling layer.
        bottleneck (nn.Sequential): Bottleneck convolutional block.
        up4 (nn.ConvTranspose2d): First up-convolution layer.
        dec4 (nn.Sequential): First decoding block.
        up3 (nn.ConvTranspose2d): Second up-convolution layer.
        dec3 (nn.Sequential): Second decoding block.
        up2 (nn.ConvTranspose2d): Third up-convolution layer.
        dec2 (nn.Sequential): Third decoding block.
        up1 (nn.ConvTranspose2d): Fourth up-convolution layer.
        dec1 (nn.Sequential): Fourth decoding block.
        final (nn.Conv2d): Final convolutional layer to produce the output.
    """

    def __init__(self, in_ch=1, out_ch=1, init_feat=32):
        """Initializes the UNetBaseline model.

        Args:
            in_ch (int, optional): Number of input channels. Defaults to 1.
            out_ch (int, optional): Number of output channels. Defaults to 1.
            init_feat (int, optional): Number of initial features. Defaults to 32.
        """
        super().__init__()

        def conv_block(ic, oc):
            """A helper function to create a convolutional block.

            Args:
                ic (int): Input channels.
                oc (int): Output channels.

            Returns:
                nn.Sequential: A sequential container of layers.
            """
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
        """Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output segmentation map.
        """
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
