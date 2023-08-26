import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataset(Dataset):
    """
    This class implement the general dataset of train, validation and testing data
    :param dataframe_file: The file path to a csv file containing data
    :param image_dir: The file path to the folder containing the images that are specified in the csv file
    :param transform: The function to transform the input images (e.g.:- resize, crop and convert to tensor)
    :param target_transform: The function to transform the label (e.g.:- convert to tensor)
    """

    def __init__(self, dataframe_file, image_dir, transform=None, target_transform=None):
        self.dataframe = pd.read_csv(dataframe_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_1_path = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        img_2_path = os.path.join(self.image_dir, self.dataframe.iloc[idx, 1])
        img_1 = read_image(img_1_path)
        img_2 = read_image(img_2_path)
        label = self.dataframe.iloc[idx, 2]
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.target_transform:
            label = self.target_transform(label)
        return img_1, img_2, label
