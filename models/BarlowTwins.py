import torch
import torch.nn as nn


class Projector(nn.Module):
    """A projector network for Barlow Twins.

    This module projects the output of an encoder to a lower-dimensional space.

    Attributes:
        layers (nn.Sequential): The sequential layers of the projector.
        gap (nn.AdaptiveAvgPool2d): Global average pooling layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initializes the Projector.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
        """
        super(Projector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """Forward pass of the Projector.

        Args:
            x (torch.Tensor): The input tensor from the encoder.

        Returns:
            torch.Tensor: The projected tensor.
        """
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.layers(x)


class BarlowTwinsModel(nn.Module):
    """A model for Barlow Twins self-supervised learning.

    This model combines an encoder and a projector for Barlow Twins.

    Attributes:
        encoder (nn.Module): The encoder network.
        projector (Projector): The projector network.
    """

    def __init__(self, encoder, projector):
        """Initializes the BarlowTwinsModel.

        Args:
            encoder (nn.Module): The encoder network.
            projector (Projector): The projector network.
        """
        super(BarlowTwinsModel, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        """Forward pass of the BarlowTwinsModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projection from the projector network.
        """
        bottleneck, _ = self.encoder.forward_encoder(x)
        projection = self.projector(bottleneck)
        return projection
