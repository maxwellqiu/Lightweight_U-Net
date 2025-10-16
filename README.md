# Lightweight U-Net for Breast Ultrasound Image Segmentation

> Wenyang Qiu, Ethan Hamburg, Yisca Perez, Yinkun Zhou, Soorena Salari, and Yaser Esmaeili Salehani

Concordia University, Montréal, Canada  
Email: wenyang.qiu@concordia.ca

## Project Overview

This repository contains the official implementation of the paper "Lightweight U-Net for Breast Ultrasound Image Segmentation". The project focuses on providing a computationally efficient and accurate deep learning model for segmenting breast lesions in ultrasound images, which is crucial for early breast cancer diagnosis.

The core of this repository is a lightweight U-Net architecture that leverages a MobileNetV2 backbone as its encoder. This design significantly reduces the model's parameter count and computational complexity (GFLOPs) without compromising its ability to extract rich, hierarchical features from ultrasound images. The model's effectiveness is demonstrated on the BUS-BRA dataset, where it achieves state-of-the-art performance, outperforming standard U-Net and Attention U-Net models.

### Key Features:

*   **Lightweight Architecture:** A MobileNetV2-based U-Net with only 3.10M parameters and 0.72 GFLOPs.
*   **High Accuracy:** Achieves a Dice coefficient of 88.06% and a mean Intersection over Union (mIoU) of 80.34%.
*   **Efficient Inference:** Fast inference times, with an average latency of 1.44 seconds per image.
*   **Self-Supervised Learning:** Includes implementations for Barlow Twins pre-training.
*   **Advanced Techniques:** Incorporates spatial attention, Squeeze-and-Excitation blocks, and multi-scale fusion for enhanced performance.

## Setup and Installation

To get started with this project, follow these steps to set up your environment and install the necessary dependencies.

### Prerequisites

*   Python 3.11.9 or later
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **Install the required packages:**

    A `requirements.txt` file is not provided, but you can install the necessary packages using pip:

    ```bash
    pip install kagglehub pandas numpy opencv-python matplotlib torch torchvision albumentations scikit-learn
    ```

## Usage

This section provides a guide on how to use the models and utilities in this repository.

### Training a Model

The `utils/training.py` module contains functions for training the models. Here's a basic example of how to train the `UNetBaseline` model:

```python
import torch
from torch.utils.data import DataLoader
from models.UNet import UNetBaseline
from utils.datasets import BusbraDataset
from utils.metric import BCEDiceLoss
from utils.training import train

# 1. Set up your device, model, and dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetBaseline(in_ch=1, out_ch=1).to(device)
# Assume 'train_df' and 'val_df' are pandas DataFrames with file paths
# and 'train_transform' and 'val_transform' are Albumentations transforms
train_dataset = BusbraDataset(df=train_df, img_size=(256, 256), transform=train_transform)
val_dataset = BusbraDataset(df=val_df, img_size=(256, 256), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 2. Define your loss function, optimizer, and scheduler
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

# 3. Start the training process
history, best_iou, best_dice, best_prec = train(
    model=model,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=50,
    save_path='best_unet_model.pth'
)

print(f"Training complete. Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")
```

### Visualizing Predictions

The `utils/visualization.py` module provides functions to help you visualize the model's performance.

#### Comparing Predictions with Ground Truth

You can use the `predict_compare` function to visually inspect the model's segmentation results against the ground truth masks.

```python
from utils.visualization import predict_compare

# Load your best model
model.load_state_dict(torch.load('best_unet_model.pth'))

# Visualize predictions on the validation set
predict_compare(model, device, val_loader, num_samples=5)
```

#### Plotting Training History

The `plot_history_loss` function can be used to plot the training and validation loss curves over epochs.

```python
from utils.visualization import plot_history_loss

# Plot the loss history from the training process
plot_history_loss(history)
```

## Abstract

Accurate segmentation of breast lesions in ultrasound images is essential for early breast cancer diagnosis and treatment planning. We propose a lightweight U-Net architecture that replaces standard convolutional encoder blocks with a MobileNetV2 backbone trained from scratch, reducing model complexity to 3.10 M parameters and 0.72 GFLOPs, while retaining rich feature extraction. Evaluated on the BUS-BRA dataset, our model outperformed the standard U-Net, Attention U-Net, and self-supervised learning models, achieving a Dice coefficient of 88.06% and a mean Intersection over Union (mIoU) of 80.34%, with an average inference latency of 1.44 ± 0.41 s per image. These results highlight the effectiveness of combining a lightweight encoder with attention mechanisms to achieve both high accuracy and computational efficiency in breast ultrasound segmentation.

**Keywords**: Breast Cancer Segmentation, U-Net, Lightweight Networks, Ultrasound Imaging, and Medical Image Analysis.