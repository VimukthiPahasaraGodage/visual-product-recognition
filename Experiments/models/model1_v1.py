from enum import Enum

import numpy as np
from torch import nn

from models.modeling import VisionTransformer, CONFIGS


class DistanceMeasures(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2


class Model1(nn.Module):
    def __init__(self, vit_model="ViT-L_16", linear_layer_output_dim=2048, distance_measure=DistanceMeasures.COSINE):
        super(Model1, self).__init__()

        config = CONFIGS[vit_model]
        model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)

        embedding_dim = 1024
        if vit_model == "ViT-L_16":
            model.load_from(np.load("C:/UoM/Semester 5/CS3501_Data_Science_and_Engineering_Project/visual-product-recognition/Experiments/model_checkpoints/ViT-L_16-224.npz"))
        elif vit_model == "ViT-B_16":
            embedding_dim = 768
            model.load_from(np.load("C:/UoM/Semester 5/CS3501_Data_Science_and_Engineering_Project/visual-product-recognition/Experiments/model_checkpoints/ViT-L_16-224.npz"))
        else:
            raise Exception("No such ViT model available.")
        self.encoder = nn.Sequential(*[model.transformer.embeddings, model.transformer.encoder])

        self.linear = nn.Linear(embedding_dim, linear_layer_output_dim)

        if distance_measure == DistanceMeasures.COSINE:
            self.distance = nn.CosineSimilarity()
        elif distance_measure == DistanceMeasures.EUCLIDEAN:
            self.distance = nn.PairwiseDistance(p=2.0)
        elif distance_measure == DistanceMeasures.MANHATTAN:
            self.distance = nn.PairwiseDistance(p=1.0)
        else:
            raise Exception("No such distance measurement exist.")

    def forward(self, img1, img2):
        embeddings_1, att_weights_1 = self.encoder(img1)  # embeddings:[batch_size, n_tokens, embedding_dim]
        embeddings_2, att_weights_2 = self.encoder(img2)
        embedding_cls_token_1 = embeddings_1[:, 0, :]  # [batch_size, embedding_dim]
        embedding_cls_token_2 = embeddings_2[:, 0, :]
        vector_1 = self.linear(embedding_cls_token_1)  # [batch_size, embedding_dim * 2]
        vector_2 = self.linear(embedding_cls_token_2)
        l2_normalized_vector_1 = nn.functional.normalize(vector_1, p=2.0, dim=1)  # [batch_size, embedding_dim * 2]
        l2_normalized_vector_2 = nn.functional.normalize(vector_2, p=2.0, dim=1)
        output = self.distance(l2_normalized_vector_1, l2_normalized_vector_2)  # [batch_size, 1]
        return output
