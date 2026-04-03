"""
Loss functions and evaluation metrics.

From: "Lightweight MobileNetV2-UNet for Breast Ultrasound
Image Segmentation" (Qiu et al., 2025)

Loss: Hybrid BCE + Dice (equal weights, squared denominator)
Metrics: Dice coefficient, Precision, mean IoU (thresholds 0.5-0.95)
"""

import torch
import torch.nn as nn
import numpy as np


class BCEDiceLoss(nn.Module):
    """Hybrid loss combining BCE and Dice with squared denominator.

    The BCE term uses BCEWithLogitsLoss for numerical stability on
    raw logits. The Dice term squares predicted probabilities in the
    denominator, which balances pixel-level and region-level gradients.

    Args:
        weight_bce (float): Weight for the BCE component. Default: 0.5.
        weight_dice (float): Weight for the Dice component. Default: 0.5.
        smooth (float): Smoothing constant to avoid division by zero.
    """

    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.wb = weight_bce
        self.wd = weight_dice
        self.smooth = smooth

    def forward(self, logits, target):
        """Compute hybrid BCE + Dice loss.

        Args:
            logits (torch.Tensor): Raw model output (before sigmoid).
            target (torch.Tensor): Binary ground truth mask.

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        bce_l = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        prob_flat = prob.view(prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + self.smooth) / (
            (prob_flat ** 2).sum(1) + (target_flat ** 2).sum(1) + self.smooth)
        dice_l = 1 - dice.mean()
        return self.wb * bce_l + self.wd * dice_l


# Multi-threshold mIoU thresholds (0.5 to 0.95 in steps of 0.05)
MIOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def compute_metrics(logits, targets):
    """Compute Dice, Precision, and multi-threshold mIoU.

    All metrics are computed from sigmoid-activated outputs.
    Dice and Precision use a fixed threshold of 0.5.
    mIoU averages IoU scores over thresholds from 0.5 to 0.95.

    Args:
        logits (torch.Tensor): Raw model output (B, 1, H, W).
        targets (torch.Tensor): Binary ground truth (B, 1, H, W).

    Returns:
        tuple: (dice, precision, miou) as floats.
    """
    probs = torch.sigmoid(logits)

    # Dice and Precision at threshold 0.5
    bp = (probs > 0.5).float()
    tp = (bp * targets).sum((1, 2, 3))
    fp = (bp * (1 - targets)).sum((1, 2, 3))
    fn = ((1 - bp) * targets).sum((1, 2, 3))
    dice = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    prec = (tp / (tp + fp + 1e-6)).mean().item()

    # Multi-threshold mIoU
    ious = []
    for t in MIOU_THRESHOLDS:
        bp_t = (probs > t).float()
        tp_t = (bp_t * targets).sum((1, 2, 3))
        fp_t = (bp_t * (1 - targets)).sum((1, 2, 3))
        fn_t = ((1 - bp_t) * targets).sum((1, 2, 3))
        ious.append((tp_t / (tp_t + fp_t + fn_t + 1e-6)).mean().item())

    return dice, prec, np.mean(ious)
