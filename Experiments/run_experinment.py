from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Experiments.models.model1_v1 import Model1
from dataset import CustomDataset
from loss_functions import cosine_similarity_contrastive_loss
from loss_functions import euclidean_manhattan_contrastive_loss
from models.model1_v1 import DistanceMeasures

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Model parameters
vit_model = "ViT-L_16"
linear_layer_output_dim = 2048
distance_measure = DistanceMeasures.COSINE

# Control parameters
freeze_vit = True
load_from_saved_model = False
saved_model_path = ''

# File paths to training, validation, and test datasets
training_dataset = '/training_dataset.csv'
validation_dataset = '/validation_dataset.csv'
test_dataset = '/test_dataset.csv'

# File paths to image directories for training/validation and testing
train_validation_image_dir = '/train_validation_images'
test_image_dir = '/test_images'

# Dataloader parameters
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 12}

# Create dataloaders
training_set = CustomDataset(training_dataset, train_validation_image_dir)
training_generator = DataLoader(training_set, **params)

validation_set = CustomDataset(validation_dataset, train_validation_image_dir)
validation_generator = DataLoader(validation_set, **params)

test_set = CustomDataset(test_dataset, test_image_dir)
test_generator = DataLoader(test_set, **params)

# Instantiate the model
if not load_from_saved_model:
    model = Model1(vit_model, linear_layer_output_dim, distance_measure)
else:
    checkpoint = torch.load(saved_model_path)
    model = Model1(vit_model, linear_layer_output_dim, distance_measure)
    model.load_state_dict(checkpoint['model_state_dict'])

non_frozen_parameters = None
if freeze_vit:
    for name, param in model.named_parameters():
        if param.requires_grad and 'encoder' in name:
            param.requires_grad = False
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
else:
    for name, param in model.named_parameters():
        param.requires_grad = True
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]

# Initialize the optimizer
optimizer = torch.optim.SGD(non_frozen_parameters, lr=0.001, momentum=0.9)

# Define the loss function
loss_fn = None
if distance_measure == DistanceMeasures.COSINE:
    loss_fn = cosine_similarity_contrastive_loss
else:
    loss_fn = euclidean_manhattan_contrastive_loss


# The Training Loop
# -----------------
#
# Below, we have a function that performs one training epoch. It
# enumerates data from the DataLoader, and on each pass of the loop does
# the following:
#
# - Gets a batch of training data from the DataLoader
# - Zeros the optimizer’s gradients
# - Performs an inference - that is, gets predictions from the model for an input batch
# - Calculates the loss for that set of predictions vs. the labels on the dataset
# - Calculates the backward gradients over the learning weights
# - Tells the optimizer to perform one learning step - that is, adjust the model’s
#   learning weights based on the observed gradients for this batch, according to the
#   optimization algorithm we chose
# - It reports on the loss for every 1000 batches.
# - Finally, it reports the average per-batch loss for the last
#   1000 batches, for comparison with a validation run
#
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for idx, data in enumerate(training_generator):
        # Every data instance is an input + label pair
        img_1, img_2, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        distances = model(img_1, img_2)

        # Compute the loss and its gradients
        loss = loss_fn(distances, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if idx % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(idx + 1, last_loss))
            tb_x = epoch_index * len(training_generator) + idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Per-Epoch Activity
# ~~~~~~~~~~~~~~~~~~
#
# There are a couple of things we’ll want to do once per epoch:
#
# - Perform validation by checking our relative loss on a set of data that was not
#   used for training, and report this
# - Save a copy of the model
#
# Here, we’ll do our reporting in TensorBoard. This will require going to
# the command line to start TensorBoard, and opening it in another browser
# tab.
#

# Initializing the summary writer and getting the current time stamp for filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(
    'runs/{}_model1_{}_{}_{}'.format(timestamp, vit_model, linear_layer_output_dim, distance_measure))

EPOCHS = 100

epoch_number = 0
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_generator):
            vimg_1, vimg_2, label = vdata
            vdistance = model(vimg_1, vimg_2)
            vloss = loss_fn(vdistance, label)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
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
        model_path = '{}_model_{}_{}_{}_{}'.format(
            timestamp, epoch_number, vit_model, linear_layer_output_dim, distance_measure)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_vloss
        }, model_path)

    epoch_number += 1
