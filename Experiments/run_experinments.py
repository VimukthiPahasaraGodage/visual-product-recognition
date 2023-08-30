from experiment import Experiment
from models.model1_v1 import DistanceMeasures
from models.model1_v1 import VitModels

training_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/train_dataset_tiniest.csv'
validation_dataset_path = '/home/group15/VPR/Project_Code/Data Preprocessing/generated_datasets/validation_dataset_tiniest.csv'
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
            use_lr_scheduling=True,
            learning_rate=0.1,
            lr_reduce_factor=0.1,
            lr_patience=2,
            lr_cooldown=0,
            batch_size=4,
            num_epochs=1)

        # Results of Experiment 2
        # ~~~~~~~~~~~~~~~~~~~~~~~

        #####################################################################################################
    case default:
        experiment = None

if experiment:
    experiment.train()
    experiment.test()
