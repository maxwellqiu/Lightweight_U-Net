import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class BusbraDataset(Dataset):

    def __init__(self, df, img_size, transform=None):
        self.df = df
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx]['image_path'],
                           cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros(self.img_size, dtype=np.uint8)
        mask = cv2.imread(self.df.iloc[idx]['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask = (mask > 0).astype('uint8')
        aug = self.transform(image=image, mask=mask)
        return aug['image'], aug['mask'].unsqueeze(0).float()


class MobileNetV2Dataset(Dataset):

    def __init__(self, df, img_size, transform=None):
        self.df = df
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
        mask = cv2.imread(self.df.iloc[idx]['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask = (mask > 0).astype('uint8')
        aug = self.transform(image=image, mask=mask)
        return aug['image'], aug['mask'].unsqueeze(0).float()


class BarlowTwinDataset(Dataset):

    def __init__(self, dataframe, img_size, transform=None):
        self.dataframe = dataframe
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_path']
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            image = np.zeros(self.img_size, dtype=np.uint8)

        view1, view2 = None, None
        try:
            view1 = self.transform(image=image)['image']
            view2 = self.transform(image=image)['image']
        except:
            view1 = torch.zeros((1, self.img_size[0], self.img_size[1]),
                                dtype=torch.float32)
            view2 = torch.zeros((1, self.img_size[0], self.img_size[1]),
                                dtype=torch.float32)

        return view1, view2
