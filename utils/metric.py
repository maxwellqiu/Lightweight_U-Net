import torch
import torch.nn as nn
import numpy as np


class BCEDiceLoss(nn.Module):
    """A loss function that combines BCE and Dice loss.

    This loss function is a weighted sum of Binary Cross-Entropy and Dice loss.

    Attributes:
        bce (nn.BCEWithLogitsLoss): The BCE loss component.
        wb (float): The weight for the BCE loss.
        wd (float): The weight for the Dice loss.
    """

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        """Initializes the BCEDiceLoss.

        Args:
            weight_bce (float, optional): The weight for the BCE loss. Defaults to 0.5.
            weight_dice (float, optional): The weight for the Dice loss. Defaults to 0.5.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.wb, self.wd = weight_bce, weight_dice

    def forward(self, logits, target):
        """Forward pass of the BCEDiceLoss.

        Args:
            logits (torch.Tensor): The model's predictions (logits).
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The combined BCE and Dice loss.
        """
        bce_l = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        prob_flat = prob.view(prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) +
                                     1e-6)
        dice_l = 1 - dice.mean()
        return self.wb * bce_l + self.wd * dice_l


class BarlowTwinsLoss(nn.Module):
    """The Barlow Twins loss function.

    This loss function encourages the cross-correlation matrix between the
    outputs of two identical networks fed with distorted versions of a sample
    to be close to the identity matrix.

    Attributes:
        lambda_param (float): The weight of the redundancy reduction term.
        batch_size (int): The batch size.
        bn (nn.BatchNorm1d): The batch normalization layer.
    """

    def __init__(self, lambda_param, batch_size, projector_output_dim):
        """Initializes the BarlowTwinsLoss.

        Args:
            lambda_param (float): The weight of the redundancy reduction term.
            batch_size (int): The batch size.
            projector_output_dim (int): The output dimension of the projector.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(projector_output_dim, affine=False)

    def forward(self, z1, z2):
        """Forward pass of the BarlowTwinsLoss.

        Args:
            z1 (torch.Tensor): The output of the first network.
            z2 (torch.Tensor): The output of the second network.

        Returns:
            torch.Tensor: The Barlow Twins loss.
        """

        if z1.shape[0] < 2 or z2.shape[0] < 2:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)

        c = z1_norm.T @ z2_norm
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.fill_diagonal_(0).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag

        return loss
