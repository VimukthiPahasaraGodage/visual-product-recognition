from enum import Enum

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Experiments.datasets import TestDataset
from Experiments.models.model1_v1 import DistanceMeasures
from Experiments.models.model1_v1 import Model1
from Experiments.models.model1_v1 import VitModels
from Experiments.transform_functions import transformations


class AvailableModels(Enum):
    Models1 = 0


class DeploymentModel:
    def __init__(self,
                 model_weights_path,
                 gallery_dataset,
                 gallery_image_dir,
                 model=AvailableModels.Models1,
                 vit_model=VitModels.ViT_L_16,
                 distance_measure=DistanceMeasures.COSINE,
                 linear_layer_output_dim=2048,
                 load_from_saved_model=True):

        self.gallery_dataset = gallery_dataset
        self.gallery_image_dir = gallery_image_dir

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        checkpoint = torch.load(model_weights_path)
        self.model = None
        if model == AvailableModels.Models1:
            self.model = Model1(vit_model=vit_model,
                                distance_measure=distance_measure,
                                linear_layer_output_dim=linear_layer_output_dim,
                                load_from_saved_model=load_from_saved_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Use gpu for model training if available
        self.model.to(self.device)
        # Enable evaluation mode in model
        self.model.eval()

    def get_sorted_gallery_images(self,
                                  query_img,
                                  query_img_dir,
                                  bbox_x,
                                  bbox_y,
                                  bbox_w,
                                  bbox_h,
                                  product_id=0):

        gallery_set = TestDataset(query_img,
                                  product_id,
                                  bbox_x,
                                  bbox_y,
                                  bbox_w,
                                  bbox_h,
                                  self.gallery_dataset,
                                  query_img_dir,
                                  self.gallery_image_dir,
                                  transformations['testing_transformation_1'])

        device = self.device

        gallery_generator = DataLoader(gallery_set, batch_size=100, shuffle=False, num_workers=12)

        distances = torch.tensor([[0]]).cpu()

        with torch.no_grad():
            for idx, data in enumerate(gallery_generator):
                query, gallery_img, label = data
                query, gallery_img = query.to(device), gallery_img.to(device)
                dist = torch.unsqueeze(self.model(query, gallery_img), dim=1)
                dist = dist.cpu()
                distances = torch.cat((distances, dist), dim=0)

        distances = distances[1:]
        _, sort_indices = torch.sort(distances, dim=0, descending=True)

        gallery_dataset = pd.read_csv(self.gallery_dataset)
        gallery_image_list = gallery_dataset['img'].to_list()
        sort_indices_list = torch.squeeze(sort_indices).tolist()
        sorted_gallery_images = [gallery_image_list[i] for i in sort_indices_list]
        return sorted_gallery_images
