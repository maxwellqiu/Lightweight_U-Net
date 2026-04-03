# Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation

> Wenyang Qiu, Ethan Hamburg, Yinkun Zhou, and Yaser Esmaeili Salehani

Concordia University, Montréal, Canada  
Email: wenyang.qiu@concordia.ca

## Overview

This repository contains the official implementation of **"Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation."** The proposed model replaces the standard U-Net encoder with a MobileNetV2 backbone pretrained on ImageNet and pairs it with a compact decoder built from depthwise-separable convolutions. A controlled ablation study shows that popular bottleneck additions (SE attention, transformer layers, multi-scale fusion) do not improve accuracy on these datasets, confirming that the pretrained encoder alone provides sufficient feature capacity.

### Key Results

| Dataset | Model | Dice (%) | mIoU (%) | Params (M) |
|---------|-------|----------|----------|------------|
| BUS-BRA | Base U-Net | 87.44 ± 0.51 | 76.22 ± 0.75 | 7.76 |
| BUS-BRA | Attention U-Net | 87.54 ± 0.38 | 76.06 ± 0.42 | 7.76 |
| BUS-BRA | ResNet18-UNet | 89.09 ± 0.51 | 79.16 ± 0.27 | 12.69 |
| BUS-BRA | **MobileNetV2-UNet (ours)** | **89.40 ± 0.15** | 78.92 ± 0.58 | **2.38** |
| BUSI | Base U-Net | 67.65 ± 0.60 | 55.06 ± 0.21 | 7.76 |
| BUSI | Attention U-Net | 73.22 ± 1.22 | 58.71 ± 0.79 | 7.76 |
| BUSI | ResNet18-UNet | **76.98 ± 0.85** | **64.50 ± 0.62** | 12.69 |
| BUSI | **MobileNetV2-UNet (ours)** | 74.89 ± 1.76 | 61.40 ± 1.04 | **2.38** |

All results are 5-run averages under identical hyperparameters, loss functions, and data splits.

### Highlights

- **2.38M parameters, 0.46 GFLOPs** — 5.3× smaller than ResNet18-UNet, 3.3× smaller than standard U-Net
- **89.40% Dice on BUS-BRA** — highest among all four models tested
- **Statistically comparable to ResNet18-UNet** (p=0.219) at a fraction of the size
- **Negative ablation result**: SE, Transformer, and MSF modules add parameters but not accuracy
- **Two-dataset evaluation**: BUS-BRA (1,875 images) and BUSI (647 images)

## Datasets

This study uses two publicly available breast ultrasound datasets:

- **BUS-BRA** (1,875 images): [Kaggle link](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset)
- **BUSI** (647 benign + malignant images): [Kaggle link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Both datasets are split 80/20 using stratified sampling on pathology label.

## Model Architecture

The proposed MobileNetV2-UNet consists of:

- **Encoder**: MobileNetV2 pretrained on ImageNet (all layers fine-tuned)
- **Bottleneck**: 1×1 convolution reducing 1280 channels to 96
- **Decoder**: 3 transposed-convolution upsampling stages with depthwise-separable convolution blocks
- **Skip connections**: Tapped at MobileNetV2 layers {1, 3, 6, 13} with {16, 24, 32, 96} channels
- **Regularization**: Dropout (p=0.3) after bottleneck
- **Output**: 1×1 convolution producing single-channel logit map

## Training Configuration

All models share the same training setup:

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=1e-4, weight_decay=5e-4) |
| Batch size | 8 |
| Epochs | 80 |
| Loss | Hybrid BCE + Dice (equal weights) |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.1) |
| Runs | 5 independent runs per model per dataset |
| GPU | NVIDIA V100 |

## Setup and Installation

### Prerequisites

- Python 3.11 or later
- CUDA-compatible GPU

### Installation

```bash
git clone https://github.com/maxwellqiu/Lightweight_U-Net.git
cd Lightweight_U-Net
pip install kagglehub pandas numpy opencv-python matplotlib torch torchvision albumentations scikit-learn
```

## Usage

### Training

```python
import torch
from torch.utils.data import DataLoader
from models.UNet import UNetBaseline
from utils.datasets import BusbraDataset
from utils.metric import BCEDiceLoss
from utils.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetBaseline(in_ch=1, out_ch=1).to(device)

train_dataset = BusbraDataset(df=train_df, img_size=(256, 256), transform=train_transform)
val_dataset = BusbraDataset(df=val_df, img_size=(256, 256), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)

history, best_iou, best_dice, best_prec = train(
    model=model,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=80,
    save_path='best_model.pth'
)
```

### Visualizing Predictions

```python
from utils.visualization import predict_compare, plot_history_loss

model.load_state_dict(torch.load('best_model.pth'))
predict_compare(model, device, val_loader, num_samples=5)
plot_history_loss(history)
```

## Ablation Study

Tested on BUS-BRA. All variants share the same encoder and decoder; only the bottleneck changes.

| Configuration | Dice (%) | mIoU (%) | Params (M) |
|---------------|----------|----------|------------|
| **Base (proposed)** | **89.40 ± 0.15** | **78.92 ± 0.58** | **2.38** |
| Base + SE | 88.52 ± 0.59 | 78.40 ± 0.69 | 2.38 |
| Base + SE + Trans | 89.05 ± 0.20 | 78.51 ± 0.33 | 2.81 |
| Base + SE + Trans + MSF | 88.57 ± 0.20 | 77.95 ± 0.42 | 3.10 |

None of the additions improve over the base model.

## Citation

If you find this work useful, please cite:

```bibtex
@article{qiu2025lightweight,
  title={Lightweight MobileNetV2-UNet for Breast Ultrasound Image Segmentation},
  author={Qiu, Wenyang and Hamburg, Ethan and Zhou, Yinkun and Esmaeili Salehani, Yaser},
  year={2025}
}
```

## License

This project is for academic research purposes.
