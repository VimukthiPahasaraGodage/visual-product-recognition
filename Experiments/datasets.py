import os

import pandas as pd
import torchvision.transforms.functional
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomDataset(Dataset):
    def __init__(self, dataframe_file, image_dir, transform=None, target_transform=None):
        """
        This class implement the general data retriever for train, validation and testing datasets

        :param dataframe_file: The file path to a csv file containing data
        :param image_dir: The file path to the folder containing the images that are specified in the csv file
        :param transform: The function to transform the input images (e.g.:- resize, crop and convert to tensor)
        :param target_transform: The function to transform the label (e.g.:- convert to tensor)
        """
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


class TestDataset(Dataset):
    def __init__(self,
                 query_img,
                 query_img_id,
                 bbox_x,
                 bbox_y,
                 bbox_w,
                 bbox_h,
                 dataframe_file,
                 query_img_dir,
                 gallery_img_dir,
                 transform=None):
        """
        This class implement the data retriever for testing dataset for mAP calculation

        :param query_img: The file path of query image
        :param query_img_id: The id associated with query image to recognize GTPs
        :param bbox_x: The x-coordinate of the bounding box
        :param bbox_y: The y-coordinate of the bounding box
        :param bbox_w: The width of the bounding box
        :param bbox_h: The height of the bounding box
        :param dataframe_file: The file path to a csv file containing data
        :param query_img_dir: The folder path which query image is located
        :param gallery_img_dir: The folder path which the gallery images are located
        :param transform: The transformation function for the images
        """
        self.dataframe = pd.read_csv(dataframe_file)
        self.gallery_img_dir = gallery_img_dir
        self.transform = transform
        self.query_img_id = query_img_id

        query_img_path = os.path.join(query_img_dir, query_img)
        query_img = read_image(query_img_path)
        cropped_query_img = torchvision.transforms.functional.crop(query_img, bbox_y, bbox_x, bbox_h, bbox_w)
        self.transformed_query_img = transform(cropped_query_img)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        gallery_img_path = os.path.join(self.gallery_img_dir, self.dataframe.iloc[idx, 0])
        gallery_img_id = self.dataframe.iloc[idx, 1]
        gallery_img = read_image(gallery_img_path)
        if self.transform:
            gallery_img = self.transform(gallery_img)
        label = 0
        if self.query_img_id == gallery_img_id:
            label = 1
        return self.transformed_query_img, gallery_img, label
