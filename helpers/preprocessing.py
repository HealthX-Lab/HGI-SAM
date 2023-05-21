from torchvision import transforms
import torch
from monai.transforms import Compose, RandFlipD, RandAffineD, RandGaussianNoiseD
from numpy import deg2rad


def window_image(image, window_params, intercept, slope, rescale=True):
    """
    A method to window CT image

    :param image: image intensities of Hounsfield units
    :param window_params: (window-center, window-width)
    :param intercept: the intercept of window
    :param slope: the slope of window
    :param rescale: whether to do a 0-1 normalization at the end
    :return: float tensor of windowed image
    """
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
    """
    preprocessing transforms applied to image, which only includes resizing the image
    to delete some parts of background, we resize it to 110% of needed size, and then we center crop the image

    :param image_size: the size of image needed
    :return: transformed image
    """
    def transform(tensor, interpolation='bilinear'):
        if interpolation == 'nearest':
            r = transforms.Resize(int(1.1 * image_size), interpolation=transforms.InterpolationMode.NEAREST)
        else:
            r = transforms.Resize(int(1.1 * image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        t = transforms.Compose([r, transforms.CenterCrop(image_size)])
        return t(tensor)
    return transform


class Augmentation:
    def __init__(self, with_mask=False):
        """
        an Augmentation class which includes random flipping, random affine transformation, and random Gaussian noise addition.

        :param with_mask: whether to apply the transforms on images masks as well.
        """
        keys = ['image', 'mask'] if with_mask else ['image']
        mode = ['bilinear', 'nearest'] if with_mask else 'bilinear'
        self.augmentation = Compose([
            RandFlipD(prob=0.5, spatial_axis=1, keys=keys),
            RandAffineD(prob=0.5, rotate_range=(deg2rad(45), deg2rad(45)), translate_range=(0.1, 0.1), scale_range=(0.1, 0.1), padding_mode='zeros', mode=mode, keys=keys),
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
