import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters import threshold_otsu


class ConfusionMatrix:
    def __init__(self, device='cuda'):
        self.device = device
        self.predictions = []
        self.ground_truth = []

        self.losses = []
        self.dices = []
        self.number_of_samples = 0

        self.true_positives = None
        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None

    def add_prediction(self, pred, gt):
        self.predictions.extend(list(pred))
        self.ground_truth.extend(list(gt))
        self.number_of_samples += len(gt)

    def add_loss(self, loss):
        self.losses.append(loss)

    def get_mean_loss(self):
        return sum(self.losses) / self.number_of_samples

    def add_dice(self, dice):
        self.dices.append(dice)

    def get_mean_dice(self):
        return sum(self.dices) / len(self.dices)

    def compute_confusion_matrix(self):
        self.predictions = torch.stack(self.predictions).to(self.device)
        self.ground_truth = torch.stack(self.ground_truth).to(self.device)

        pred_out = torch.round(self.predictions)
        self.true_positives = torch.sum(pred_out * self.ground_truth, dim=0)
        self.true_negatives = torch.sum((1 - pred_out) * (1 - self.ground_truth), dim=0)
        self.false_positives = torch.sum(pred_out * (1 - self.ground_truth), dim=0)
        self.false_negatives = torch.sum((1 - pred_out) * self.ground_truth, dim=0)

    def get_accuracy(self):
        numerator = self.true_positives + self.true_negatives
        denominator = numerator + self.false_positives + self.false_negatives

        return torch.divide(numerator, denominator)

    def get_precision(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives

        return torch.divide(numerator, denominator)

    def get_recall_sensitivity(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives

        return torch.divide(numerator, denominator)

    def get_specificity(self):
        numerator = self.true_negatives
        denominator = self.true_negatives + self.false_positives

        return torch.divide(numerator, denominator)

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall_sensitivity()
        numerator = 2 * precision * recall
        denominator = precision + recall

        return torch.divide(numerator, denominator)

    def get_auc_score(self):
        scores = []
        for i in range(self.ground_truth.shape[-1]):
            scores.append(metrics.roc_auc_score(self.ground_truth[:, 0, i].cpu().numpy(),
                                                self.predictions[:, 0, i].cpu().numpy()))
        return np.array(scores)


class FocalBCELoss:
    def __init__(self, gamma=2, reduction='sum'):
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, gt):
        sigmoid = torch.sigmoid(pred)
        bce = -(gt * ((1 - sigmoid) ** self.gamma) * torch.log(sigmoid) +
                (1 - gt) * (sigmoid ** self.gamma) * torch.log(1 - sigmoid))

        if self.reduction == 'sum':
            reduced_bce = bce.sum(dim=0)
        else:
            reduced_bce = bce.mean(dim=0)

        return reduced_bce


class EarlyStopping:
    def __init__(self, model: nn.Module, patience: int, path_to_save: str, gamma=0):
        assert patience > 0, 'patience must be positive'
        self.model = model
        self.patience = patience
        self.gamma = gamma
        self.path_to_save = path_to_save

        self.min_loss = np.Inf
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss + self.gamma < self.min_loss:
            print("val loss decreased from {} to {}".format(self.min_loss, val_loss))
            self.min_loss = val_loss
            self.counter = 0
            save_model(self.model, self.path_to_save)
        else:
            save_model(self.model, self.path_to_save + r'early.pt')
            self.counter += 1
            if self.counter == self.patience:
                print('early stop')
                self.early_stop = True
            else:
                print('early stopping counter: {} of {}'.format(self.counter, self.patience))


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))


def plot_diagram(*metrics):
    fig, axes = plt.subplots(len(metrics))
    for i in range(len(metrics)):
        axes[i].plot(range(1, 1 + len(metrics[i])), metrics[i])
    plt.show()


def dice_metric(predicted_mask, gt_mask):
    p = torch.count_nonzero(predicted_mask > 0)
    gt = torch.count_nonzero(gt_mask > 0)
    if p + gt == 0:
        return torch.tensor([1])
    else:
        overlap = torch.count_nonzero((predicted_mask * gt_mask) > 0)
        return 2 * overlap / (p + gt)


def intersection_over_union(predicted_mask, gt_mask):
    intersection = torch.count_nonzero(predicted_mask * gt_mask)
    union = torch.count_nonzero(predicted_mask + gt_mask)
    return intersection / union


def binarization_simple_thresholding(image, threshold):
    t = threshold * image.max()
    image[image < t] = 0
    image[image >= t] = 1


def binarization_otsu(image):
    blured_image = cv2.GaussianBlur(image.cpu().numpy(), (33, 33), cv2.BORDER_DEFAULT)
    uint_image = (blured_image * 256).astype("uint8")

    t = threshold_otsu(uint_image, 256)
    image[uint_image < t] = 0
    image[uint_image >= t] = 1
    return image
