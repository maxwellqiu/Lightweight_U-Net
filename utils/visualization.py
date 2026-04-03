"""
Visualization utilities for segmentation results.

From: "Lightweight MobileNetV2-UNet for Breast Ultrasound
Image Segmentation" (Qiu et al., 2025)
"""

import torch
import matplotlib.pyplot as plt


def predict_compare(model, device, dataloader, num_samples=3):
    """Display model predictions alongside ground truth masks.

    Shows input image, ground truth mask, and predicted mask
    side by side for visual inspection.

    Args:
        model (nn.Module): Trained segmentation model.
        device (torch.device): Device to run inference on.
        dataloader (DataLoader): Validation data loader.
        num_samples (int): Number of samples to display. Default: 3.
    """
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            if count >= num_samples:
                break
            imgs = imgs.to(device)
            logits = model(imgs)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            imgs_np = imgs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            pred_np = pred.cpu().numpy()

            for i in range(min(num_samples - count, imgs.size(0))):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Handle both 1-channel and 3-channel images
                if imgs_np[i].shape[0] == 3:
                    axes[0].imshow(imgs_np[i][0], cmap='gray')
                else:
                    axes[0].imshow(imgs_np[i][0], cmap='gray')
                axes[0].axis('off')
                axes[0].set_title('Input')

                axes[1].imshow(masks_np[i][0], cmap='gray')
                axes[1].axis('off')
                axes[1].set_title('Ground Truth')

                axes[2].imshow(pred_np[i][0], cmap='gray')
                axes[2].axis('off')
                axes[2].set_title('Prediction')

                plt.tight_layout()
                plt.show()
                plt.close(fig)
                count += 1


def plot_history_loss(history):
    """Plot training and validation loss curves.

    Args:
        history (dict): Dictionary with 'train_loss' and 'val_loss' lists.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], '.', label='Train Loss')
    plt.plot(epochs, history['val_loss'], '.', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history_metrics(history):
    """Plot validation Dice, mIoU, and Precision curves.

    Args:
        history (dict): Dictionary with 'val_dice', 'val_miou',
                        and 'val_precision' lists.
    """
    epochs = range(1, len(history['val_dice']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['val_dice'], label='Dice')
    plt.plot(epochs, history['val_miou'], label='mIoU')
    plt.plot(epochs, history['val_precision'], label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
