import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from rwightman_files.swin_transformer import window_reverse
import cv2
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters import threshold_otsu


class WeightedBCELoss:
    def __init__(self, pos_neg_weights, subtypes_weights, reduction='sum'):
        self.pos_neg_weights = pos_neg_weights.T
        self.subtypes_weights = subtypes_weights
        self.reduction = reduction

    def __call__(self, pred, gt):
        sigmoid = torch.sigmoid(pred)
        bce = -(self.pos_neg_weights[1] * gt * torch.log(sigmoid) +
                self.pos_neg_weights[0] * (1 - gt) * torch.log(1 - sigmoid))

        if self.reduction == 'sum':
            reduced_bce = bce.sum(dim=0)
        else:
            reduced_bce = bce.mean(dim=0)
        return (reduced_bce * self.subtypes_weights).mean()


class FocalBCELoss:
    def __init__(self, subtypes_weights, gamma=5, reduction='sum'):
        self.subtypes_weights = subtypes_weights
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

        return (reduced_bce * self.subtypes_weights).mean()


class WeightedFocalBCELoss:
    def __init__(self, pos_neg_weights, subtypes_weights, gamma=4, reduction='sum'):
        self.pos_neg_weights = pos_neg_weights.T
        self.subtypes_weights = subtypes_weights
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, gt):
        sigmoid = torch.sigmoid(pred)
        bce = -(self.pos_neg_weights[1] * gt * ((1 - sigmoid) ** self.gamma) * torch.log(sigmoid) +
                self.pos_neg_weights[0] * (1 - gt) * (sigmoid ** self.gamma) * torch.log(1 - sigmoid))

        if self.reduction == 'sum':
            reduced_bce = bce.sum(dim=0)
        else:
            reduced_bce = bce.mean(dim=0)

        return (reduced_bce * self.subtypes_weights).mean()


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


class ConfusionMatrix:
    def __init__(self, device='cuda'):
        self.device = device
        self.predictions = []
        self.ground_truth = []
        self.dices = []
        self.true_positives = None
        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None

    def __call__(self, pred, gt):
        self.predictions.append(pred)
        self.ground_truth.append(gt)

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


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))


def get_per_class_accuracy(pred, gt):
    predictions = torch.round(torch.sigmoid(pred))
    return (predictions == gt).sum(dim=0) / len(gt)


def plot_diagram(*metrics):
    fig, axes = plt.subplots(len(metrics))
    for i in range(len(metrics)):
        axes[i].plot(range(1, 1 + len(metrics[i])), metrics[i])
    plt.show()


def get_attentions(att_dict, layer_block):
    def hook(model, inp, out):
        att_dict[layer_block] = out.detach()

    return hook


def get_attention_mask(attentions, g_cam=None, img_size=384, patch_size=4, window_size=12, device='cpu'):
    func = _get_layer_attention_mask
    softmax_layer = nn.Softmax(dim=1)

    mask0 = func(attentions, img_size, patch_size, window_size, 0, list(range(2)), device)
    mask1 = func(attentions, img_size, patch_size, window_size, 1, list(range(2)), device)
    mask2 = func(attentions, img_size, patch_size, window_size, 2, list(range(18)), device)
    mask3 = func(attentions, img_size, patch_size, window_size, 3, list(range(2)), device)
    # cv2.imshow('last', mask3.cpu().numpy())
    # cv2.imshow('second last', mask2.cpu().numpy())

    if g_cam is None:
        f_mask = mask3 * mask2 * mask1 * mask0
    else:
        g_cam = torch.tensor(g_cam, device=device)
        f_mask = g_cam * mask2 * mask1 * mask0

    H, W = f_mask.shape
    softmax = f_mask
    softmax = softmax.reshape(1, -1)
    softmax = softmax_layer(softmax)
    softmax = softmax.reshape(H, W)
    softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())
    # cv2.imwrite(r'C:\rsna-ich\sample images\softmax.bmp', 255 * softmax.cpu().numpy())

    return softmax


def _get_layer_attention_mask(attentions: dict, img_size: int, patch_size: int, window_size: int,
                              layer: int, blocks: list, device='cpu'):
    total_tokens = img_size // patch_size
    layer_tokens = total_tokens // (2 ** layer)
    shift_size = window_size // 2

    avg_pooling = nn.AdaptiveAvgPool2d((1, window_size * window_size))
    masks = []
    for block_index in blocks:
        att = attentions[f'{layer}_{block_index}']
        att = torch.mean(att, dim=1)  # mean over heads
        mask = avg_pooling(att).reshape(-1, window_size * window_size)  # pool over target sequences

        mask = mask.reshape(-1, window_size, window_size, 1)
        mask = window_reverse(mask, window_size, layer_tokens, layer_tokens)
        # if layer == 0:
        #     print(mask.shape)
        #     x = mask.squeeze().cpu().numpy()
        #     x = (x - x.min()) / (x.max() - x.min())
        #     cv2.imwrite(rf'C:\rsna-ich\sample images\maps\w{block_index}.bmp', x * 255)
        if block_index % 2 == 1:  # reverse shift for odd blocks
            mask = torch.roll(mask, shifts=(shift_size, shift_size), dims=(1, 2))
            # if layer == 0:
            #     x = mask.squeeze().cpu().numpy()
            #     x = (x - x.min()) / (x.max() - x.min())
            #     cv2.imwrite(rf'C:\rsna-ich\sample images\maps\sw{block_index}.bmp', x * 255)
        mask = mask.permute(0, 3, 1, 2)
        masks.append(mask)

    final_masks = []
    for i in range(0, len(masks), 2):
        swin_block_mask = masks[i] * masks[i + 1]
        # if layer == 0:
        #     x = swin_block_mask.squeeze().cpu().numpy()
        #     x = (x - x.min()) / (x.max() - x.min())
        #     cv2.imwrite(r'C:\rsna-ich\sample images\maps\mult.bmp', x * 255)
        final_masks.append(swin_block_mask)

    all_masks = torch.stack(final_masks)
    final_mask = torch.mean(all_masks, dim=0)

    final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())

    # cv2.imwrite(rf'C:\rsna-ich\sample images\layer {layer} attention.bmp', 255 * final_mask.squeeze().cpu().numpy())
    final_mask = F.interpolate(final_mask, size=(img_size, img_size), mode='bilinear').squeeze()
    # cv2.imwrite(rf'C:\rsna-ich\sample images\layer {layer} interpolated.bmp', 255 * final_mask.cpu().numpy())
    return final_mask


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