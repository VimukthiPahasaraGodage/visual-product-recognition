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
        return functional.pad(image, padding, 0, 'constant')


transformations = {
    'train_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomApply([transforms.RandomRotation((0, 45))], p=0.1),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.1),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.05),
        transforms.ToTensor()
    ]),
    'validation_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ]),
    'testing_transformation_1': transforms.Compose([
        SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
}

target_transformations = {
    'cosine_distance_transform': lambda x: -1 if x == 0 else 1
}
