import torch


def cosine_similarity_contrastive_loss(output, target):
    """
    This function returns the contrastive loss of two embeddings based on cosine similarity
    :param output: model output [batch size, 1]
    :param target: target values (Similar = 1; dissimilar = -1) [batch size, 1]
    :return: mean contrastive loss of batch; value range =  [0, 1]
    """
    loss_contrastive = torch.mean(torch.add(torch.multiply(torch.sub(1, target), torch.add(1, output)),
                                            torch.multiply(torch.add(1, target), torch.sub(1, output))))
    loss_contrastive = torch.div(loss_contrastive, 4)
    return loss_contrastive


def euclidean_manhattan_contrastive_loss(output, target, margin=1.0):
    """
    This function returns the contrastive loss of two embeddings based on Euclidean or Manhattan distance
    :param output: model output [batch size, 1]
    :param target: target values (Similar = 1; dissimilar = 0) [batch size, 1]
    :param margin: margin distance between two dissimilar images (default = 1.0)
    :return: mean contrastive loss of batch
    """
    loss_contrastive = torch.mean(torch.add(torch.multiply(target, torch.square(output)),
                                            torch.multiply(torch.sub(1, target),
                                                           torch.clamp(torch.pow(margin - output, 2), min=0))))
    loss_contrastive = torch.div(loss_contrastive, 2)
    return loss_contrastive
