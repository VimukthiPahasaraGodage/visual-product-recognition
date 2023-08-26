import os
from urllib.request import urlretrieve

os.makedirs("model_checkpoints", exist_ok=True)  # make a folder to save model checkpoints
# Download and save the model checkpoints for ViT-L_16-224
if not os.path.isfile("model_checkpoints/ViT-L_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz",
                "model_checkpoints/ViT-L_16-224.npz")

# Download and save the model checkpoints for ViT-B_16-224
if not os.path.isfile("model_checkpoints/ViT-B_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
                "model_checkpoints/ViT-B_16-224.npz")
