from torchvision import transforms
import torch


def window_image(image, window_center_width, rescale=True):
    img_min = window_center_width[0] - window_center_width[1] // 2
    img_max = window_center_width[0] + window_center_width[1] // 2
    image[image < img_min] = img_min
    image[image > img_max] = img_max

    if rescale and (image.max() - image.min()) > 0:
        image = (image - image.min()) / (image.max() - image.min())

    return image


class _AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cuda'):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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
