"""
Dataset classes for breast ultrasound segmentation.

From: "Lightweight MobileNetV2-UNet for Breast Ultrasound
Image Segmentation" (Qiu et al., 2025)

Supports BUS-BRA and BUSI datasets.
"""

import numpy as np
import cv2
from torch.utils.data import Dataset


class BusbraDataset(Dataset):
    """Dataset class for BUS-BRA breast ultrasound images.

    Reads grayscale ultrasound images, converts to 3-channel RGB
    (by replicating the grayscale values), and binarizes masks.

    Args:
        df (pd.DataFrame): DataFrame with 'image_path' and 'mask_path' columns.
        img_size (tuple): Target image size (H, W).
        transform (callable, optional): Albumentations transform pipeline.
    """

    def __init__(self, df, img_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gray = cv2.imread(self.df.iloc[idx]['image_path'],
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros(self.img_size, dtype=np.uint8)
        image = np.stack([gray] * 3, axis=-1)

        mask = cv2.imread(self.df.iloc[idx]['mask_path'],
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask = (mask > 0).astype('uint8')

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            return aug['image'], aug['mask'].unsqueeze(0).float()

        return image, mask


class BusiDataset(Dataset):
    """Dataset class for BUSI breast ultrasound images.

    BUSI images are natively RGB, so no channel conversion is needed.
    Masks are binarized (nonzero -> 1).

    Args:
        df (pd.DataFrame): DataFrame with 'image_path' and 'mask_path' columns.
        img_size (tuple): Target image size (H, W).
        transform (callable, optional): Albumentations transform pipeline.
    """

    def __init__(self, df, img_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.df.iloc[idx]['image_path'])
        if img is None:
            img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.df.iloc[idx]['mask_path'],
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask = (mask > 0).astype('uint8')

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            return aug['image'], aug['mask'].unsqueeze(0).float()

        return img, mask
