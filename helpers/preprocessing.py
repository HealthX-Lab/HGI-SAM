from torchvision import transforms
import torch
import numpy as np
import random
from monai.transforms import Compose, RandFlipD, RandAffineD, RandGaussianNoiseD
from numpy import deg2rad


def window_image(image, window_params, intercept, slope, rescale=True):
    center, width = window_params
    img_min = center - width // 2
    img_max = center + width // 2

    image = image * slope + intercept
    image[image < img_min] = img_min
    image[image > img_max] = img_max

    if rescale and (image.max() - image.min()) > 0:
        image = (image - image.min()) / (image.max() - image.min())

    return torch.FloatTensor(image)


def get_transform(image_size):
    t = transforms.Compose([
        transforms.Resize(int(1.1 * image_size)),
        transforms.CenterCrop(image_size)
    ])

    return t


class Augmentation:
    def __init__(self, with_mask=False):
        keys = ['image', 'mask'] if with_mask else ['image']
        self.augmentation = Compose([
            RandFlipD(prob=0.5, spatial_axis=1, keys=keys),
            RandAffineD(prob=0.5, rotate_range=(deg2rad(45), deg2rad(45)), translate_range=(0.1, 0.1), scale_range=(0.1, 0.1), padding_mode='zeros', keys=keys),
            RandGaussianNoiseD(prob=0.5, keys=['image'])
        ])

    def __call__(self, image, mask=None):
        if mask is not None:
            x = {'image': image, 'mask': mask}
        else:
            x = {'image': image}

        x = self.augmentation(x)
        if mask is not None:
            return x['image'], x['mask']
        else:
            return x['image']
