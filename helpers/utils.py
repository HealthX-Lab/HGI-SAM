import torch.nn as nn
import torch
import numpy as np
import cv2
from sklearn import metrics
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from monai.transforms.post.array import one_hot
from typing import List, cast
from torch import Tensor, einsum


def simplex(t: Tensor, axis=1) -> bool:
    """
    A helper function used to make sure tensors are onehot like
    All rights reserved to: https://github.com/LIVIAETS/boundary-loss
    """
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        """
        A class to compute Cross Entropy Loss.
        All rights reserved to: https://github.com/LIVIAETS/boundary-loss
        """
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        super().__init__()
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum("bkxy,bkxy->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        A class to compute Dice Loss.
        All rights reserved to: https://github.com/LIVIAETS/boundary-loss
        """
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        super().__init__()
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcxy,bcxy->bc", pc, tc)
        union: Tensor = (einsum("bkxy->bk", pc) + einsum("bkxy->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class ConfusionMatrix:
    def __init__(self, device='cuda'):
        """
        A class built to save predictions, store model performance metrics, and carry out confusion matrix operations

        :param device: device where to do the torch computations
        """
        self.device = device
        self.predictions = []
        self.ground_truth = []

        self.losses = []
        self.number_of_samples = 0

        self.true_positives = None
        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None

        self.dices = []
        self.IoUs = []
        self.hausdorff_distances = []

    def add_prediction(self, pred, gt):
        self.predictions.extend(list(pred.detach()))
        self.ground_truth.extend(list(gt.detach()))

    def add_number_of_samples(self, new_samples):
        self.number_of_samples += new_samples

    def add_loss(self, loss):
        self.losses.append(loss)

    def get_mean_loss(self):
        return sum(self.losses) / self.number_of_samples

    def add_dice(self, dice):
        self.dices.extend(list(dice))

    def add_iou(self, iou):
        self.IoUs.extend(list(iou))

    def add_hausdorff_distance(self, hd):
        self.hausdorff_distances.extend(list(hd))

    def get_mean_dice(self):
        return torch.nanmean(torch.tensor(self.dices))

    def get_mean_iou(self):
        return torch.nanmean(torch.tensor(self.IoUs))

    def get_mean_hausdorff_distance(self):
        return torch.nanmean(torch.tensor(self.hausdorff_distances))

    def compute_confusion_matrix(self):
        self.predictions = torch.tensor(self.predictions).to(self.device)
        self.ground_truth = torch.tensor(self.ground_truth).to(self.device)

        self.true_positives = torch.sum(self.predictions * self.ground_truth, dim=0)
        self.true_negatives = torch.sum((1 - self.predictions) * (1 - self.ground_truth), dim=0)
        self.false_positives = torch.sum(self.predictions * (1 - self.ground_truth), dim=0)
        self.false_negatives = torch.sum((1 - self.predictions) * self.ground_truth, dim=0)

    def get_accuracy(self):
        numerator = self.true_positives + self.true_negatives
        denominator = numerator + self.false_positives + self.false_negatives

        return torch.divide(numerator, denominator).item()

    def get_precision(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives

        return torch.divide(numerator, denominator).item()

    def get_recall_sensitivity(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives

        return torch.divide(numerator, denominator).item()

    def get_specificity(self):
        numerator = self.true_negatives
        denominator = self.true_negatives + self.false_positives

        return torch.divide(numerator, denominator).item()

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall_sensitivity()
        numerator = 2 * precision * recall
        denominator = precision + recall

        return torch.divide(numerator, denominator).item()

    def get_auc_score(self):
        # scores = []
        # for i in range(self.ground_truth.shape[-1]):
        #     scores.append(metrics.roc_auc_score(self.ground_truth[:, 0, i].cpu().numpy(),
        #                                         self.predictions[:, 0, i].cpu().numpy()))
        # return np.array(scores)
        return metrics.roc_auc_score(self.ground_truth.cpu().numpy(), self.predictions.cpu().numpy())


class DiceCELoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        A combo loss class at combines Cross Entropy and Dice Loss

        :param alpha: ratio of the Cross Entropy Loss in the final loss
        """
        super().__init__()
        assert alpha <= 1
        self.ce_loss = CrossEntropy(idc=[0, 1])
        self.dice_loss = DiceLoss(idc=[0, 1])
        self.alpha = alpha

    def forward(self, pred_mask, target_mask):
        target_mask = one_hot(target_mask, 2, dim=1)
        return self.alpha * self.ce_loss(pred_mask, target_mask) + (1 - self.alpha) * self.dice_loss(pred_mask, target_mask)


class EarlyStopping:
    def __init__(self, model: nn.Module, patience: int, path_to_save: str, gamma=0):
        """
        A class that is used to track the validation loss of a training and stop it if it does not decrease

        :param model: pytorch model
        :param patience: for how many epochs we continue training until early stopping
        :param path_to_save: saving path for the best performing model on validation data so far.
        :param gamma: the least difference that validation loss must decrease
        """
        assert patience > 0, 'patience must be positive'
        self.model = model
        self.patience = patience
        self.path_to_save = path_to_save
        self.gamma = gamma

        self.min_loss = np.Inf
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss + self.gamma < self.min_loss:
            print("val loss decreased from {} to {}".format(self.min_loss, val_loss))
            self.min_loss = val_loss
            self.counter = 0
            save_model(self.model, f'{self.path_to_save}')
        else:
            self.counter += 1
            if self.counter == self.patience:
                print("early stop")
                self.early_stop = True
            else:
                print("early stopping counter: {} of {}".format(self.counter, self.patience))


def save_model(model: nn.Module, path: str):
    """
    a method save weights of a model

    :param model: pytorch model
    :param path: saving path of the state dictionary of the model
    """
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str):
    """
    a method to load weights of a model

    :param model: pytorch model
    :param path: path to state dictionary of the model
    """
    model.load_state_dict(torch.load(path))


def binarization_simple_thresholding(image, threshold):
    """
    a method that applies a simple thresholding on a 2D image and binarizes it

    :param image: input image (prediction map)
    :param threshold: threshold value
    :return: binary image
    """
    image[image < threshold] = 0
    image[image >= threshold] = 1
    return image


def binarization_otsu(image):
    """
    a method that applies an OTSU thresholding on a 2D image and binarizes it

    :param image: input image (prediction map)
    :return: binary image
    """
    binarized_image = torch.zeros_like(image)
    for index in range(binarized_image.shape[0]):
        im = image[index]
        blured_image = cv2.GaussianBlur(im.cpu().numpy(), (33, 33), cv2.BORDER_DEFAULT)
        uint_image = (blured_image * 256).astype("uint8")

        t = threshold_otsu(uint_image, 256)
        im[uint_image < t] = 0
        im[uint_image >= t] = 1

        binarized_image[index] = im
    return binarized_image


def str_to_bool(string):
    """
    Helper method that converts a string into boolean

    :param string: either True or False
    """
    return True if string == "True" else False


def visualize_losses(train_losses, valid_losses, path_to_save):
    """
    A method used to plot the train and validation losses of a training, which are showed in two rows respectively.

    :param train_losses: list of train losses
    :param valid_losses: list of validation losses
    :param path_to_save: path to save the figure
    """
    fig, axes = plt.subplots(2)
    axes[0].plot(train_losses)
    axes[1].plot(valid_losses)
    plt.savefig(path_to_save)


def reshape_transform(tensor):
    """
    A helper method to reshape the tensors used in GradCAM approach
    """
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result


def to_onehot(tensor):
    """
    A helper method to convert a binary tensor into a onehot tensor

    :param tensor: input binary tensor
    :return: onehot tensor
    """
    onehot = torch.zeros(2, *tensor.shape[1:], device=tensor.device)
    onehot[1] = tensor
    onehot[0] = 1 - tensor
    return onehot
