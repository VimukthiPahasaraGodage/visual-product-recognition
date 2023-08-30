from datetime import datetime
from enum import Enum

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Experiments.models.model1_v1 import Model1
from accuracy_functions import average_precision
from accuracy_functions import mean_average_precision
from datasets import CustomDataset
from datasets import TestDataset
from loss_functions import cosine_similarity_contrastive_loss
from loss_functions import euclidean_manhattan_contrastive_loss
from models.model1_v1 import DistanceMeasures
from models.model1_v1 import VitModels
from transform_functions import target_transformations
from transform_functions import transformations


class OptimizersType(Enum):
    SGD = 0
    Adam = 1


class Experiment:
    def __init__(self,
                 experiment_name,
                 training_dataset,
                 validation_dataset,
                 query_dataset,
                 gallery_dataset,
                 train_validation_image_dir,
                 query_image_dir,
                 gallery_image_dir,
                 vit_model=VitModels.ViT_L_16,
                 linear_layer_output_dim=2048,
                 distance_measure=DistanceMeasures.COSINE,
                 freeze_vit=False,
                 load_from_saved_model=False,
                 load_from_saved_optim_state=False,
                 saved_model_path='',
                 optimizer_type=OptimizersType.Adam,
                 learning_rate=0.001,
                 use_lr_scheduling=False,
                 lr_reduce_factor=0.1,
                 lr_patience=10,
                 lr_cooldown=5,
                 batch_size=32,
                 shuffle=True,
                 num_epochs=100,
                 ):
        """
        Initialize an experiment with a model

        :param experiment_name: Name of the experiment.
        :param training_dataset: Training dataset path.
        :param validation_dataset: Validation dataset path.
        :param query_dataset: Path of dataset containing information about queries.
        :param gallery_dataset: Path of dataset containing information about gallery images.
        :param train_validation_image_dir: Folder path where images of training and validation images are located.
        :param query_image_dir: Folder path where query images of test images are located.
        :param gallery_image_dir: Folder path where gallery images of test images are located
        :param vit_model: The variation of the ViT model to be used in the experiment.
        :param linear_layer_output_dim: Number of output neurons in the linear layer.
        :param distance_measure: Cosine distance, Euclidean distance or Manhattan Distance.
        :param freeze_vit: Lock the weights of the ViT encoder.
        :param load_from_saved_model: Initialize model weights from a previously saved model.
        :param load_from_saved_optim_state: Restore the state of optimizer from a previously saved model.
        :param saved_model_path: The file path of the saved model to be used for model weights and optimizer state.
        :param optimizer_type: The type of optimizer, SGD or Adam.
        :param learning_rate: The initial learning rate of the optimizer.
        :param use_lr_scheduling: Use ReduceLROnPlateau learning rate scheduler.
        :param lr_reduce_factor: The factor of ReduceLROnPlateau.
        :param batch_size: Batch size for DataLoader.
        :param shuffle: Make True to shuffle the data when generating batches in DataLoader.
        :param num_epochs: Number of epochs to train.
        """
        self.experiment_name = experiment_name
        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
        self.query_image_dir = query_image_dir
        self.gallery_image_dir = gallery_image_dir
        self.vit_model = vit_model
        self.linear_layer_output_dim = linear_layer_output_dim
        self.distance_measure = distance_measure
        self.freeze_vit = freeze_vit
        self.load_from_saved_model = load_from_saved_model
        self.load_from_saved_optim_state = load_from_saved_optim_state
        self.saved_model_path = saved_model_path
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.use_lr_scheduling = use_lr_scheduling
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_patience = lr_patience
        self.lr_cooldown = lr_cooldown
        self.num_epochs = num_epochs

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        # Dataloader parameters
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': 12}

        # Make the dissimilar label -1 from 0 when using cosine similarity as distance measurement
        if distance_measure.value == DistanceMeasures.COSINE.value:
            _target_transform = target_transformations['cosine_distance_transform']
        else:
            _target_transform = None

        # Create dataloaders
        training_set = CustomDataset(training_dataset, train_validation_image_dir,
                                     transform=transformations['train_transformation_1'],
                                     target_transform=_target_transform)
        self.training_generator = DataLoader(training_set, **params)

        validation_set = CustomDataset(validation_dataset, train_validation_image_dir,
                                       transform=transformations['validation_transformation_1'],
                                       target_transform=_target_transform)
        self.validation_generator = DataLoader(validation_set, **params)

        # Initialize the model
        self.model = self.__initialize_model()

        # Use gpu for model training if available
        self.model.to(self.device)

        # freeze the ViT model or unfreeze all the layers of the model
        non_frozen_parameters = self.__freeze_unfreeze_layers(self.model)

        # Initialize the optimizer and restore the optimizer state if specified
        self.optimizer = self.__initialize_optimizer(non_frozen_parameters)

        # Initialize the learning rate scheduler
        if self.use_lr_scheduling:
            self.scheduler = self.__initialize_learning_rate_scheduler(self.optimizer)

        # Initialize the loss function
        self.loss_fn = self.__initialize_loss_function()

        # Initialize the summary writer
        self.writer = self.__initialize_summary_writer()

    def __initialize_model(self):
        if not self.load_from_saved_model:
            model = Model1(self.vit_model,
                           self.linear_layer_output_dim,
                           self.distance_measure,
                           load_from_saved_model=False)
        else:
            checkpoint = torch.load(self.saved_model_path)
            model = Model1(self.vit_model,
                           self.linear_layer_output_dim,
                           self.distance_measure,
                           load_from_saved_model=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def __freeze_unfreeze_layers(self, model):
        if self.freeze_vit:
            for name, param in model.named_parameters():
                if param.requires_grad and 'encoder' in name:
                    param.requires_grad = False
            non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
            non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        return non_frozen_parameters

    def __initialize_optimizer(self, non_frozen_parameters):
        if self.optimizer_type.value == OptimizersType.Adam.value:
            optimizer = torch.optim.Adam(non_frozen_parameters, lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(non_frozen_parameters, lr=self.learning_rate, momentum=0.9)
        if self.load_from_saved_optim_state:
            checkpoint = torch.load(self.saved_model_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer

    def __initialize_learning_rate_scheduler(self, optimizer):
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode='min',
                                         factor=self.lr_reduce_factor,
                                         patience=self.lr_patience,
                                         cooldown=self.lr_cooldown,
                                         min_lr=0.00001)
        return lr_scheduler

    def __initialize_loss_function(self):
        if self.distance_measure.value == DistanceMeasures.COSINE.value:
            loss_fn = cosine_similarity_contrastive_loss
        else:
            loss_fn = euclidean_manhattan_contrastive_loss
        return loss_fn

    def __initialize_summary_writer(self):
        # Getting the current time stamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(
            'runs/{}_experiment.{}_model1_vit_model.{}_outdim.{}_distm.{}_optim.{}'.format(timestamp,
                                                                                           self.experiment_name,
                                                                                           self.vit_model,
                                                                                           self.linear_layer_output_dim,
                                                                                           self.distance_measure,
                                                                                           self.optimizer_type))
        return writer

    def __train_one_epoch(self, device, epoch_index, tb_writer):
        """
        The Training Loop
        -----------------

        Below, we have a function that performs one training epoch. It
        enumerates data from the DataLoader, and on each pass of the loop does
        the following:

        - Gets a batch of training data from the DataLoader
        - Zeros the optimizer’s gradients
        - Performs an inference - that is, gets predictions from the model for an input batch
        - Calculates the loss for that set of predictions vs. the labels on the dataset
        - Calculates the backward gradients over the learning weights
        - Tells the optimizer to perform one learning step - that is, adjust the model’s
          learning weights based on the observed gradients for this batch, according to the
          optimization algorithm we chose
        - It reports on the loss for every 1000 batches.
        - Finally, it reports the average per-batch loss for the last
          1000 batches, for comparison with a validation run

        :param epoch_index:
        :param tb_writer:
        :return:
        """
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for idx, data in enumerate(self.training_generator):
            # Every data instance is an input + label pair
            img_1, img_2, labels = data

            img_1, img_2, labels = img_1.to(device), img_2.to(device), labels.to(device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            distances = self.model(img_1, img_2)

            # Compute the loss and its gradients
            loss = self.loss_fn(distances, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if idx % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch_index * len(self.training_generator) + idx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self):
        """
        Per-Epoch Activity
        ~~~~~~~~~~~~~~~~~~

        There are a couple of things we’ll want to do once per epoch:

        - Perform validation by checking our relative loss on a set of data that was not
          used for training, and report this
        - Save a copy of the model

        Here, we’ll do our reporting in TensorBoard. This will require going to
        the command line to start TensorBoard, and opening it in another browser
        tab.

        :return:
        """
        # The device used for model inference(CUDA)
        device = self.device

        EPOCHS = self.num_epochs

        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(self.device, epoch_number, self.writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_generator):
                    vimg_1, vimg_2, label = vdata
                    vimg_1, vimg_2, label = vimg_1.to(device), vimg_2.to(device), label.to(device)
                    vdistance = self.model(vimg_1, vimg_2)
                    vloss = self.loss_fn(vdistance, label)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            if self.use_lr_scheduling:
                self.scheduler.step(avg_vloss)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch_number + 1)
            self.writer.flush()

            # Track the best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'saved.{}_experiment.{}_model1_epoch.{}_vit_model.{}_outdim.{}_distm.{}_optim.{}' \
                    .format(datetime.now().strftime('%Y%m%d_%H%M%S'),
                            self.experiment_name,
                            epoch_number,
                            self.vit_model,
                            self.linear_layer_output_dim,
                            self.distance_measure,
                            self.optimizer_type)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'val_loss': avg_vloss
                }, model_path)

            epoch_number += 1

    def test(self):
        # The device used for model inference(CUDA)
        device = self.device

        # Read query dataframe
        queries = pd.read_csv(self.query_dataset, low_memory=False)

        # Enable evaluation mode in model
        self.model.eval()

        query = 0
        average_precisions = torch.tensor([0])

        for index, row in queries.iterrows():
            product_id = row['id']
            test_set = TestDataset(row['img'],
                                   product_id,
                                   row['bbox_x'],
                                   row['bbox_y'],
                                   row['bbox_w'],
                                   row['bbox_h'],
                                   self.gallery_dataset,
                                   self.query_image_dir,
                                   self.gallery_image_dir,
                                   transformations['test_transformation_1'])
            test_generator = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=12)

            distances = torch.tensor([[0]])
            gtp_indices = torch.tensor([[0]])

            for idx, data in enumerate(test_generator):
                query, gallery_img, label = data
                query, gallery_img, label = query.to(device), gallery_img.to(device), label.to(device)
                dist = self.model(query, gallery_img)
                gtps = torch.sub(1, torch.abs(torch.round(torch.clamp(torch.sub(product_id, label), min=-1, max=1))))
                distances = torch.cat((distances, dist), dim=0)
                gtp_indices = torch.cat((gtp_indices, gtps), dim=0)

            distances = distances[1:]
            gtp_indices = gtp_indices[1:]
            _, sort_indices = torch.sort(distances, dim=0, descending=True)
            rearranged_gpts = torch.unsqueeze(torch.squeeze(gtp_indices[torch.add(1, sort_indices)]), 1)
            gtp_positions, _ = torch.sort(torch.squeeze(torch.multiply(rearranged_gpts, torch.add(1, sort_indices))),
                                          dim=0,
                                          descending=False)
            gtp_positions = torch.squeeze(gtp_positions[gtp_positions.nonzero()])
            avg_precision = average_precision(gtp_positions)
            print('Query {} Average Precision {}'.format(query + 1, avg_precision.item()))

            average_precisions = torch.cat((average_precisions, avg_precision), dim=0)

            # Log the average precision per query
            self.writer.add_scalars('Testing: Average Precisions',
                                    {'Query': query, 'Query_Image': row['img'], 'Validation': avg_precision.item()},
                                    query + 1)
            self.writer.flush()

            query += 1

        average_precisions = average_precisions[1:]
        mean_avg_precision = mean_average_precision(average_precisions)
        print('Mean Average Precision {}'.format(mean_avg_precision))

        # Log the mean average precision for the model
        self.writer.add_scalars('Mean Average Precision', mean_avg_precision.item())
        self.writer.flush()
