import torch
import torch.nn as nn


class Projector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.layers(x)


class BarlowTwinsModel(nn.Module):

    def __init__(self, encoder, projector):
        super(BarlowTwinsModel, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        bottleneck, _ = self.encoder.forward_encoder(x)
        projection = self.projector(bottleneck)
        return projection
