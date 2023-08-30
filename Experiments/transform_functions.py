import numpy as np
from torchvision import transforms
from torchvision.transforms import functional


class SquarePad:
    def __call__(self, image):
        c, w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return functional.pad(image, padding, 0, 'constant').numpy()


transformations = {
    'train_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomApply([transforms.RandomRotation((0, 180))], p=0.4),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
        transforms.ToTensor()
    ]),
    'validation_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    'testing_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}

target_transformations = {
    'cosine_distance_transform': lambda x: -1 if x == 0 else 1
}
