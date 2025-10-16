import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    This block recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Sequential): A sequential container of fully connected layers.
    """

    def __init__(self, channel, reduction=16):
        """Initializes the SEBlock.

        Args:
            channel (int): Number of input channels.
            reduction (int, optional): Reduction ratio for the number of channels.
                Defaults to 16.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1),
                                nn.ReLU(True),
                                nn.Conv2d(channel // reduction, channel, 1),
                                nn.Sigmoid())

    def forward(self, x):
        """Forward pass of the SEBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with channel-wise recalibration.
        """
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution.

    This module performs a depthwise convolution followed by a pointwise convolution.

    Attributes:
        dw (nn.Conv2d): Depthwise convolutional layer.
        pw (nn.Conv2d): Pointwise convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self, in_ch, out_ch):
        """Initializes the DepthwiseSeparableConv module.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
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
        """Forward pass of the DepthwiseSeparableConv module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.dw(x)
        x = self.pw(x)
        return self.relu(self.bn(x))


class TransformerBlock(nn.Module):
    """A Transformer encoder block.

    This module applies a standard Transformer encoder to a feature map.

    Attributes:
        encoder (nn.TransformerEncoder): The Transformer encoder.
    """

    def __init__(self, dim, heads=4, layers=1):
        """Initializes the TransformerBlock.

        Args:
            dim (int): The embedding dimension.
            heads (int, optional): The number of attention heads. Defaults to 4.
            layers (int, optional): The number of encoder layers. Defaults to 1.
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        """Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        b, c, h, w = x.shape
        xf = x.flatten(2).permute(2, 0, 1)
        xf = self.encoder(xf)
        return xf.permute(1, 2, 0).view(b, c, h, w)


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion module.

    This module fuses features from multiple scales using upsampling,
    convolution, and a Squeeze-and-Excitation block.

    Attributes:
        branches (nn.ModuleList): A list of branches for different scales.
        se (SEBlock): The Squeeze-and-Excitation block.
        project (nn.Conv2d): The final projection layer.
    """

    def __init__(self, in_ch, out_ch, scales=[1, 2, 4]):
        """Initializes the MultiScaleFusion module.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            scales (list, optional): A list of scaling factors. Defaults to [1, 2, 4].
        """
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
        """Forward pass of the MultiScaleFusion module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The fused output tensor.
        """
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
    """A U-Net model with a MobileNetV2 encoder, attention, and multi-scale fusion.

    This class implements a U-Net architecture that uses a pretrained MobileNetV2
    as the encoder, and includes attention and multi-scale fusion mechanisms.

    Attributes:
        img_size (tuple): The size of the input image.
        encoder (nn.Module): The MobileNetV2 encoder.
        bottle (nn.Conv2d): A bottleneck layer.
        se (SEBlock): The Squeeze-and-Excitation block.
        trans (TransformerBlock): The Transformer block.
        msf (MultiScaleFusion): The multi-scale fusion module.
        dropout (nn.Dropout2d): The dropout layer.
        refine (nn.Conv2d): A refinement layer.
        up_convs (nn.ModuleList): A list of up-convolution layers.
        dec_convs (nn.ModuleList): A list of decoder convolution layers.
        final (nn.Conv2d): The final convolutional layer.
    """

    def __init__(self, dropout_p, img_size, out_channels=1):
        """Initializes the MobileNetV2_UNet_Attn_MS model.

        Args:
            dropout_p (float): The dropout probability.
            img_size (tuple): The size of the input image.
            out_channels (int, optional): The number of output channels. Defaults to 1.
        """
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
        """Forward pass of the MobileNetV2_UNet_Attn_MS model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output segmentation map.
        """
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 6, 13]:
                skips.append(x)
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
