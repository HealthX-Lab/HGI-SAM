from torchvision import transforms
import torch


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


class _AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cuda'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_augmentation():
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([_AddGaussianNoise(0., 0.09)])
    ])

    return t
