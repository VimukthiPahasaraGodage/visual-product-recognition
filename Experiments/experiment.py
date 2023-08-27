from datetime import datetime
from enum import Enum

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Experiments.models.model1_v1 import Model1
from dataset import CustomDataset
from loss_functions import cosine_similarity_contrastive_loss
from loss_functions import euclidean_manhattan_contrastive_loss
from models.model1_v1 import DistanceMeasures
from models.model1_v1 import VitModels


class OptimizersType(Enum):
    SGD = 0
    Adam = 1


class Experiment:
    def __init__(self,
                 experiment_name,
                 training_dataset,
                 validation_dataset,
                 test_dataset,
                 train_validation_image_dir,
                 test_image_dir,
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
                 batch_size=32,
                 shuffle=True,
                 num_epochs=100):
        """

        :param experiment_name:
        :param training_dataset:
        :param validation_dataset:
        :param test_dataset:
        :param train_validation_image_dir:
        :param test_image_dir:
        :param vit_model:
        :param linear_layer_output_dim:
        :param distance_measure:
        :param freeze_vit:
        :param load_from_saved_model:
        :param load_from_saved_optim_state:
        :param saved_model_path:
        :param optimizer_type:
        :param learning_rate:
        :param use_lr_scheduling:
        :param lr_reduce_factor:
        :param batch_size:
        :param shuffle:
        :param num_epochs:
        """
        self.experiment_name = experiment_name
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
        self.num_epochs = num_epochs

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        # Dataloader parameters
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': 12}

        # Create dataloaders
        training_set = CustomDataset(training_dataset, train_validation_image_dir)
        self.training_generator = DataLoader(training_set, **params)

        validation_set = CustomDataset(validation_dataset, train_validation_image_dir)
        self.validation_generator = DataLoader(validation_set, **params)

        test_set = CustomDataset(test_dataset, test_image_dir)
        self.test_generator = DataLoader(test_set, **params)

        # Initialize the model
        self.model = self.__initialize_model()

        # Use gpu for model training if available
        self.model.to(device)

        # freeze the ViT model or unfreeze all the layers of the model
        non_frozen_parameters = self.__freeze_unfreeze_layers(self.model)

        # Initialize the optimizer and restore the optimizer state if specified
        self.optimizer = self.__initialize_optimizer(non_frozen_parameters)

        # Initialize the learning rate scheduler
        if self.use_lr_scheduling:
            self.scheduler = self.__initialize_learning_rate_scheduler(self.optimizer)

        # Initialize the loss function
        self.loss_fn = self.__initialize_loss_function()

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
        if self.optimizer_type == OptimizersType.Adam:
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
                                         patience=10,
                                         cooldown=5,
                                         min_lr=0.00001)
        return lr_scheduler

    def __initialize_loss_function(self):
        if self.distance_measure == DistanceMeasures.COSINE:
            loss_fn = cosine_similarity_contrastive_loss
        else:
            loss_fn = euclidean_manhattan_contrastive_loss
        return loss_fn

    def __train_one_epoch(self, epoch_index, tb_writer):
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

        # Initializing the summary writer and getting the current time stamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(
            'runs/{}_experiment.{}_model1_vit_model.{}_outdim.{}_distm.{}'.format(timestamp,
                                                                                  self.experiment_name,
                                                                                  self.vit_model,
                                                                                  self.linear_layer_output_dim,
                                                                                  self.distance_measure))

        EPOCHS = self.num_epochs

        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_generator):
                    vimg_1, vimg_2, label = vdata
                    vdistance = self.model(vimg_1, vimg_2)
                    vloss = self.loss_fn(vdistance, label)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            if self.use_lr_scheduling:
                self.scheduler.step(avg_vloss)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            writer.flush()

            # Track the best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = '{}_experiment.{}_model1_epoch.{}_vit_model.{}_outdim.{}_distm.{}' \
                    .format(timestamp,
                            self.experiment_name,
                            epoch_number,
                            self.vit_model,
                            self.linear_layer_output_dim,
                            self.distance_measure)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'val_loss': avg_vloss
                }, model_path)

            epoch_number += 1
