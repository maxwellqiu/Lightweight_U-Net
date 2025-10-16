import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class BusbraDataset(Dataset):
    """A dataset class for the BUS-BRA dataset.

    Attributes:
        df (pd.DataFrame): The dataframe containing image and mask paths.
        img_size (tuple): The size of the images.
        transform (callable, optional): A function/transform to apply to the images and masks.
    """

    def __init__(self, df, img_size, transform=None):
        """Initializes the BusbraDataset.

        Args:
            df (pd.DataFrame): The dataframe containing image and mask paths.
            img_size (tuple): The size to resize images to.
            transform (callable, optional): A function/transform to apply to the images and masks.
                Defaults to None.
        """
        self.df = df
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Retrieves an item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and mask tensors.
        """
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
    """A dataset class for use with MobileNetV2.

    This dataset class prepares images for MobileNetV2 by converting them to
    3-channel format.

    Attributes:
        df (pd.DataFrame): The dataframe containing image and mask paths.
        img_size (tuple): The size of the images.
        transform (callable, optional): A function/transform to apply to the images and masks.
    """

    def __init__(self, df, img_size, transform=None):
        """Initializes the MobileNetV2Dataset.

        Args:
            df (pd.DataFrame): The dataframe containing image and mask paths.
            img_size (tuple): The size to resize images to.
            transform (callable, optional): A function/transform to apply to the images and masks.
                Defaults to None.
        """
        self.df = df
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Retrieves an item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and mask tensors.
        """
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
    """A dataset class for Barlow Twins self-supervised learning.

    This dataset generates two augmented views of each image for Barlow Twins training.

    Attributes:
        dataframe (pd.DataFrame): The dataframe containing image paths.
        img_size (tuple): The size of the images.
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, dataframe, img_size, transform=None):
        """Initializes the BarlowTwinDataset.

        Args:
            dataframe (pd.DataFrame): The dataframe containing image paths.
            img_size (tuple): The size to resize images to.
            transform (callable, optional): A function/transform to apply to the images.
                Defaults to None.
        """
        self.dataframe = dataframe
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Retrieves an item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing two augmented views of the image.
        """
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
