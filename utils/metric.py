import torch
import torch.nn as nn
import numpy as np


class BCEDiceLoss(nn.Module):

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.wb, self.wd = weight_bce, weight_dice

    def forward(self, logits, target):
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

    def __init__(self, lambda_param, batch_size, projector_output_dim):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(projector_output_dim, affine=False)

    def forward(self, z1, z2):

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
