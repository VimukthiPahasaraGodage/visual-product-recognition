import torch
import numpy as np


def average_precision(gtp_positions):
    if gtp_positions.shape[0] == 0:
        return 0
    num_images = torch.tensor(np.array([i for i in range(1, gtp_positions.shape[0] + 1)]))
    fractions = torch.div(num_images, gtp_positions)
    sum_of_fractions = torch.sum(fractions)
    avg_precision = torch.div(sum_of_fractions, len(gtp_positions))
    return avg_precision


def mean_average_precision(average_precisions):
    return torch.mean(average_precisions)

