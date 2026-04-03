"""
Training and evaluation utilities.

From: "Lightweight MobileNetV2-UNet for Breast Ultrasound
Image Segmentation" (Qiu et al., 2025)

All models are trained with the same setup:
  Adam (lr=1e-4, weight_decay=5e-4), 80 epochs, batch size 8,
  ReduceLROnPlateau (patience=3, factor=0.1), checkpoint by best mIoU.
"""

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np
import torch

from utils.metric import compute_metrics


def train_epoch(model, device, train_loader, criterion, optimizer):
    """Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to train on.
        train_loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (BCEDiceLoss).
        optimizer (Optimizer): The optimizer.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, msks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(train_loader.dataset)


def eval_epoch(model, device, val_loader, criterion):
    """Evaluate the model on the validation set.

    Computes loss, Dice, Precision, and multi-threshold mIoU.
    Applies sigmoid to raw logits before metric computation.

    Args:
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to evaluate on.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function (BCEDiceLoss).

    Returns:
        tuple: (val_loss, miou, dice, precision)
    """
    model.eval()
    total_loss = 0.0
    all_dice, all_prec, all_miou = [], [], []
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            logits = model(imgs)
            loss = criterion(logits, msks)
            total_loss += loss.item() * imgs.size(0)
            dice, prec, miou = compute_metrics(logits, msks)
            all_dice.append(dice)
            all_prec.append(prec)
            all_miou.append(miou)
    val_loss = total_loss / len(val_loader.dataset)
    return val_loss, np.mean(all_miou), np.mean(all_dice), np.mean(all_prec)


def train(model,
          device,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          scheduler,
          num_epochs=80,
          save_path='best.pth'):
    """Train a model and select the best checkpoint by validation mIoU.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to train on.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function (BCEDiceLoss).
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs. Default: 80.
        save_path (str): Path to save the best model. Default: 'best.pth'.

    Returns:
        tuple: (history, best_miou, best_dice, best_precision)
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_precision': [],
        'val_miou': [],
    }
    best_miou = -np.inf
    best_dice = 0.0
    best_prec = 0.0

    pbar = tqdm(range(num_epochs), desc="Epoch")
    for epoch in pbar:
        tr_loss = train_epoch(model, device, train_loader, criterion,
                              optimizer)
        val_loss, val_miou, val_dice, val_prec = eval_epoch(
            model, device, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_precision'].append(val_prec)
        history['val_miou'].append(val_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            best_dice = val_dice
            best_prec = val_prec
            torch.save(model.state_dict(), save_path)

        scheduler.step(val_miou)
        pbar.set_description(
            f"Ep {epoch+1}/{num_epochs}  "
            f"Loss: {tr_loss:.4f}  "
            f"Dice: {val_dice:.4f}  "
            f"mIoU: {val_miou:.4f}  "
            f"Prec: {val_prec:.4f}")

    return history, best_miou, best_dice, best_prec
