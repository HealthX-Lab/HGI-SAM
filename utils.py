import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters import threshold_otsu
from monai.metrics.utils import get_mask_edges, get_surface_distance


class ConfusionMatrix:
    def __init__(self, device='cuda'):
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
        self.predictions.extend(list(pred))
        self.ground_truth.extend(list(gt))
        self.add_number_of_samples(len(gt))

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        dice_loss = self.dice(inputs, targets, smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalDiceBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.dice = DiceLoss()
        self.focal_bce = FocalLoss(gamma, alpha, reduction)

    def forward(self, inputs, targets, smooth=1):
        dice_loss = self.dice(inputs, targets, smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        focal_BCE = self.focal_bce(inputs, targets)
        Dice_Focal_BCE = BCE + dice_loss + focal_BCE

        return Dice_Focal_BCE


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.dice = DiceLoss()
        self.focal_bce = FocalLoss(gamma, alpha, reduction)

    def forward(self, inputs, targets, smooth=1):
        dice_loss = self.dice(inputs, targets, smooth)
        focal_BCE = self.focal_bce(inputs, targets)
        Dice_Focal_BCE = dice_loss + focal_BCE

        return Dice_Focal_BCE


class EarlyStopping:
    def __init__(self, model: nn.Module, patience: int, path_to_save: str, gamma=0):
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
            save_model(self.model, self.path_to_save)
        else:
            self.counter += 1
            if self.counter == self.patience:
                print("early stop")
                self.early_stop = True
            else:
                print("early stopping counter: {} of {}".format(self.counter, self.patience))


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
    p = torch.count_nonzero(predicted_mask > 0, dim=(1, 2))
    gt = torch.count_nonzero(gt_mask > 0, dim=(1, 2))
    overlap = torch.count_nonzero((predicted_mask * gt_mask) > 0, dim=(1, 2))
    return 2 * overlap / (p + gt)


def intersection_over_union(predicted_mask, gt_mask):
    intersection = torch.count_nonzero((predicted_mask * gt_mask) > 0, dim=(1, 2))
    union = torch.count_nonzero((predicted_mask + gt_mask) > 0, dim=(1, 2))
    return intersection / union


def hausdorff_distance(pred, gt):
    max_dist = np.sqrt(gt.shape[1] ** 2 + gt.shape[2] ** 2)
    distances = torch.zeros(gt.shape[0])
    for image_index in range(gt.shape[0]):
        if torch.all(torch.eq(pred[image_index], gt[image_index])):
            distances[image_index] = 0.0
            continue
        (edges_pred, edges_gt) = get_mask_edges(pred[image_index], gt[image_index])
        surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
        if surface_distance.shape == (0,):
            distances[image_index] = 0.0
            continue
        dist = surface_distance.max()
        if dist > max_dist:
            distances[image_index] = 1.0
            continue
        distances[image_index] = dist / max_dist
    return distances


def binarization_simple_thresholding(image, threshold):
    image[image < threshold] = 0
    image[image >= threshold] = 1
    return image


def binarization_otsu(image):
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
