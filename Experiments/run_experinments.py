from experiment import Experiment
from models.model1_v1 import DistanceMeasures
from models.model1_v1 import VitModels

training_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/train_dataset_small.csv'
validation_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/validation_dataset_small.csv'
query_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/test_queries.csv'
gallery_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/test_gallery.csv'
train_validation_image_dir = '/home/group15/VPR/train_validation_images'
query_image_dir = '/home/group15/VPR/query_images'
gallery_image_dir = '/home/group15/VPR/gallery_images'

experiment_number = 2
match experiment_number:
    case 1:
        # Experiment 1 ######################################################################################
        """
            - No layers are freeze
            - Not using LR scheduler
        """
        experiment = Experiment(
            'experiment1.unfreeze_layers.no_lr_schedule',
            training_dataset_path,
            validation_dataset_path,
            query_dataset_path,
            gallery_dataset_path,
            train_validation_image_dir,
            query_image_dir,
            gallery_image_dir,
            vit_model=VitModels.ViT_B_16,
            linear_layer_output_dim=768,
            num_epochs=1)

        # Results of Experiment 1
        # ~~~~~~~~~~~~~~~~~~~~~~~

        #####################################################################################################
    case 2:
        # Experiment 2 ######################################################################################
        """
            - No layers are freeze
            - Not using LR scheduler
        """
        experiment = Experiment(
            'experiment1.unfreeze_layers.no_lr_schedule',
            training_dataset_path,
            validation_dataset_path,
            query_dataset_path,
            gallery_dataset_path,
            train_validation_image_dir,
            query_image_dir,
            gallery_image_dir,
            vit_model=VitModels.ViT_B_16,
            linear_layer_output_dim=768,
            distance_measure=DistanceMeasures.EUCLIDEAN,
            learning_rate=0.01,
            batch_size=16,
            num_epochs=2)

        # Results of Experiment 2
        # ~~~~~~~~~~~~~~~~~~~~~~~

        #####################################################################################################
    case default:
        experiment = None

if experiment:
    experiment.train()
    experiment.test()
