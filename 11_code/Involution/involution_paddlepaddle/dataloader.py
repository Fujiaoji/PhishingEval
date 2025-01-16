import cv2
import os
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Normalize
from pathlib import Path
import paddle.vision.transforms as T
import pandas as pd


class TargetFeature(Dataset):
    """
    step: follow paddle.io.Dataset
    """
    def __init__(self, data_path, transform=None):
        """
        step2: __init__
        """
        super().__init__()
        # data = [] # train/test data [image_path, label]
        data_df = pd.read_csv(data_path)
        data = data_df["fileame"].tolist()
        self.data_list = data
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = cv2.imread(image_path)#, cv2.IMREAD_GRAYSCALE)

        image = image.astype('float32')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data_list)

class APWGFeature(Dataset):
    """
    step: follow paddle.io.Dataset
    """
    def __init__(self, df_data, transform=None):
        """
        step2: __init__
        """
        super().__init__()
        data = df_data["logo_path"].tolist()
        self.data_list = data
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = cv2.imread(image_path)#, cv2.IMREAD_GRAYSCALE)

        image = image.astype('float32')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data_list)
