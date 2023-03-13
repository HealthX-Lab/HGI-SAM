from torchvision import transforms
import torch
import numpy as np
import random


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


class _AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cuda'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Augmentation:
    def __init__(self, device='cuda'):
        self.displacement = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 0.9)),
            transforms.RandomRotation(45),
        ])
        self.color_change = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1),
            transforms.RandomApply([_AddGaussianNoise(0., 0.05, device)])
        ])
        self.augmentation = transforms.Compose([
            self.displacement,
            self.color_change
        ])

    def __call__(self, image, mask=None):
        seed = np.random.randint(123456789)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)

        image = self.augmentation(image)

        if mask is not None:
            random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)
            mask = self.displacement(mask)
            return image, mask
        else:
            return image
