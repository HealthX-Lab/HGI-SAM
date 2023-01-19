import pydicom
from torchvision import transforms
import torch
import numpy as np
import os


def _window_image(image, window_center_width, intercept, slope, rescale=True):
    # credit: https://www.kaggle.com/code/dcstang/see-like-a-radiologist-with-systematic-windowing/notebook
    image = (image * slope + intercept)

    if window_center_width in [-1, 0, 1]:
        if window_center_width == -1:
            image[image >= 0] = 0
            image[image < -1024] = 0
        elif window_center_width == 0:
            image[image < 0] = 0
            image[image >= 1024] = 0
        else:
            image[image < 1024] = 0
            image[image >= 2048] = 0
    else:
        img_min = window_center_width[0] - window_center_width[1] // 2
        img_max = window_center_width[0] + window_center_width[1] // 2
        image[image < img_min] = img_min
        image[image > img_max] = img_max

    if rescale and (image.max() - image.min()) > 0:
        image = (image - image.min()) / (image.max() - image.min())

    return image


def _get_first_of_dicom_field_as_int(data):
    # credit: https://www.kaggle.com/code/dcstang/see-like-a-radiologist-with-systematic-windowing/notebook
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(data) == pydicom.multival.MultiValue:
        return int(data[0])
    else:
        return int(data)


def _get_windowing(data):
    # credit: https://www.kaggle.com/code/dcstang/see-like-a-radiologist-with-systematic-windowing/notebook
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]


class _AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cuda'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_slice_image(windows: [str], file_path=None, file=None):
    WINDOWS_CENTER_WIDTH = {
        'brain': (40, 80),
        'subdural': (80, 200),
        'bone': (600, 2800),
        'soft': (40, 380),

        'brain2': (40, 120),
        'bone2': (700, 3200),
        'subdural2': (80, 280),

        'brain3': (40, 100),
        'subdural3': (80, 200),
        'bone3': (400, 500),

        'first': -1,
        'second': 0,
        'third': 1,
    }
    assert file_path is not None or file is not None, 'file path or file is needed'
    if file_path is not None:
        assert os.path.isfile(file_path), 'wrong file path'
        assert all([window in WINDOWS_CENTER_WIDTH.keys() for window in windows]), 'wrong window name'

        slice_2d = pydicom.read_file(file_path)
        window_center, window_width, intercept, slope = _get_windowing(slice_2d)
        img = torch.tensor(slice_2d.pixel_array.astype(np.float32))
    else:
        slice_2d = file
        img = torch.tensor(slice_2d.astype(np.float32))
        intercept, slope = 0, 1

    window_images = []
    for window in windows:
        window_images.append(_window_image(img, WINDOWS_CENTER_WIDTH[window], intercept, slope))

    return torch.stack(window_images)


class Transform:
    def __init__(self, img_size):
        h, w = img_size
        self.transform = transforms.Compose([
            transforms.Resize((int(h * 1.1), int(w * 1.1))),
            transforms.CenterCrop((h, w)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def update_transform(self, std_mean):
        std, mean = std_mean
        self.transform = transforms.Compose([self.transform,
                                             transforms.Normalize(mean, std)])

    def __call__(self):
        return self.transform


def get_augmentation():
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([_AddGaussianNoise(0., 0.09)])
    ])

    return t