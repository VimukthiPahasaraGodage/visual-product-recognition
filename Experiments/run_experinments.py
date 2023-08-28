from experiment import Experiment

training_dataset_path = '/hhdj'
validation_dataset_path = '/bfhbfh'
query_dataset_path = '/kkfkf'
gallery_dataset_path = '/bhdbhhdb'
train_validation_image_dir = '/jddjkj'
query_image_dir = '/njdnjd'
gallery_image_dir = '/hhdfbhdf'

# Experiment 1 ######################################################################################
"""
    - No layers are freeze
    - Not using LR scheduler
"""
experiment1 = Experiment(
    'experiment1.unfreeze_layers.no_lr_schedule',
    training_dataset_path,
    validation_dataset_path,
    query_dataset_path,
    gallery_dataset_path,
    train_validation_image_dir,
    query_image_dir,
    gallery_image_dir,
    num_epochs=1)

experiment1.train()

experiment1.test()

# Results of Experiment 1
# ~~~~~~~~~~~~~~~~~~~~~~~

#####################################################################################################
