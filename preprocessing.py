from torchvision import transforms
import torch
import numpy as np
import random


def window_image(image, window_params, rescale=True):
    center, width, intercept, slope = window_params
    img_min = center - width // 2
    img_max = center + width // 2

    image = slope * (image + intercept)
    image[image < img_min] = img_min
    image[image > img_max] = img_max

    if rescale and (image.max() - image.min()) > 0:
        image = (image - image.min()) / (image.max() - image.min())

    return torch.FloatTensor(image)


def get_transform(image_size):
    t = transforms.Compose([
        transforms.Resize(int(1.2 * image_size)),
        transforms.CenterCrop(image_size)
    ])

    return t


class Augmentation:
    def __init__(self, device='cuda'):
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 0.9)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.1, 0.1),
            transforms.GaussianBlur(3, sigma=(0.1, 1))
        ])

    def __call__(self, image, mask=None):
        seed = np.random.randint(123456789)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)

        image = self.augmentation(image)

        if mask is not None:
            random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)
            mask = self.augmentation(mask)
            return image, mask
        else:
            return image
