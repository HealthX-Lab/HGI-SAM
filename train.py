import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from utils import *
from tqdm import tqdm
from preprocessing import get_augmentation
from copy import deepcopy
import statistics
from datasets import PhysioNetICHDataset
from torchvision.transforms import GaussianBlur
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(height, width):
    def reshape(tensor):
        result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return reshape


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn,
                    train_loader, valid_loader, device='cuda', get_with_masks=False):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar_train.set_description('training')
    train_loss = []
    train_acc = []
    for i, (sample, mask, label) in pbar_train:

        optimizer.zero_grad()
        sample, label = get_augmentation()(sample.to(device)), label.to(device)
        pred = model(sample)

        loss = loss_fn(pred, label) / len(label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_acc.append(get_per_class_accuracy(pred, label))

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)

    valid_loss = []
    valid_acc = []
    if valid_loader is not None:
        model.eval()
        pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader))
        pbar_valid.set_description('validating')
        with torch.no_grad():
            for i, (sample, mask, label) in pbar_valid:
                sample, label = sample.to(device), label.to(device)
                pred = model(sample)

                loss = loss_fn(pred, label) / len(label)
                valid_loss.append(loss.item())
                valid_acc.append(get_per_class_accuracy(pred, label))
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)

    return train_loss, valid_loss, train_acc, valid_acc


def predict(model: torch.nn.Module, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    results = []

    pbar = tqdm(dataloader, total=len(dataloader))
    pbar.set_description('evaluating')
    with torch.no_grad():
        for sample in pbar:
            sample = sample.to(device)
            output = model(sample)
            pred = torch.sigmoid(output)
            results.extend(list(pred))

    return results


def segmentation(model: torch.nn.Module, dataloader, threshold, show_images=False, device='cpu'):
    model.to(device)
    model.eval()

    # gcam0 = GradCAM(model=model, target_layers=[model.layers[0].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(96, 96))
    # gcam1 = GradCAM(model=model, target_layers=[model.layers[1].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(48, 48))
    # gcam2 = GradCAM(model=model, target_layers=[model.layers[2].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(24, 24))
    # gcam3 = GradCAM(model=model, target_layers=[model.layers[3].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(12, 12))

    confusion_matrix = ConfusionMatrix()
    attentions = {}
    if isinstance(model, nn.Sequential):
        backbone = model[0]
    else:
        backbone = model
    for ln, layer in enumerate(backbone.layers):
        for bn, block in enumerate(layer.blocks):
            block.attn.softmax.register_forward_hook(get_attentions(attentions, f'{ln}_{bn}'))

    dices = []
    ious = []
    for sample, mask, label in dataloader:
        attentions.clear()
        sample, mask = sample.to(device), mask.to(device)
        with torch.no_grad():
            output = model(sample)
            confusion_matrix(torch.sigmoid(output), label)

        dice, iou = np.nan, np.nan
        if mask.any():
            img = sample.squeeze()
            brain_window = img[0]
            mask = mask[0]

            # cv2.imwrite(r'C:\rsna-ich\sample images\sample.bmp', 255 * img.permute(1, 2, 0).cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\brain window.bmp', 255 * brain_window.cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\subdural window.bmp', 255 * img[1].cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\bone window.bmp', 255 * img[2].cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\mask.bmp', 255 * mask.cpu().numpy())
            # grayscale_cam = gcam3(input_tensor=sample)[0]
            # pred_mask = torch.tensor(grayscale_cam, device=device)

            grayscale_cam = None
            pred_mask = get_attention_mask(attentions, grayscale_cam, device=device)

            denoized_brain_window = torch.tensor(cv2.GaussianBlur(brain_window.cpu().numpy(), (9, 9),
                                                                  cv2.BORDER_DEFAULT), device=device)
            # cv2.imwrite(r'C:\rsna-ich\sample images\denoised brain window.bmp',
            #             255 * denoized_brain_window.cpu().numpy())
            pred_mask = pred_mask * denoized_brain_window

            # cv2.imshow('pred mask', pred_mask.cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\pred mask.bmp', 255 * pred_mask.cpu().numpy())
            # cv2.imshow('attention map', pred_mask.cpu().numpy())

            pred_mask = binarization_simple_thresholding(pred_mask, threshold)

            # cv2.imwrite(r'C:\rsna-ich\sample images\threshold mask.bmp', 255 * pred_mask.cpu().numpy())

            dice = dice_metric(pred_mask, mask).item()
            confusion_matrix.dices.append(dice)
            # print('\n', dice)
            iou = intersection_over_union(pred_mask, mask).item()

            if show_images:
                TP = torch.where(mask > 0, 1, 0) * torch.where(pred_mask > 0, 1, 0)
                FP = torch.where(pred_mask > 0, 1, 0) * torch.where(mask > 0, 0, 1)
                FN = torch.where(mask > 0, 1, 0) * torch.where(pred_mask > 0, 0, 1)

                m = torch.stack([FN, TP, FP])
                m = m.permute(1, 2, 0).cpu().numpy().astype(np.float)

                gt_mask = deepcopy(brain_window)
                gt_mask[FN > 0] = 1

                predicted_mask = deepcopy(brain_window)
                predicted_mask[FP > 0] = 1

                correct_pred = deepcopy(brain_window)
                correct_pred[TP > 0] = 1

                segmentation_image = torch.stack([gt_mask, correct_pred, predicted_mask])
                segmentation_image = segmentation_image.permute(1, 2, 0).cpu().numpy()

                cv2.imshow('segmentation', segmentation_image)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\segmentation-combined.bmp', 255 * segmentation_image)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\brain_window.bmp', 255 * brain_window.cpu().numpy())
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\pmask-combined.bmp', 255 * m)
                # cv2.imshow('mask', m)
                #
                # grayscale_cam0 = gcam3(input_tensor=sample)[0]
                #
                # # grayscale_cam1 = gcam1(input_tensor=sample)[0]
                # # grayscale_cam2 = gcam2(input_tensor=sample)[0]
                # # grayscale_cam3 = gcam3(input_tensor=sample)[0]
                #
                # cv2.imshow('cam0', grayscale_cam0)
                # # cv2.imshow('cam1', grayscale_cam1)
                # # cv2.imshow('cam2', grayscale_cam2)
                # # cv2.imshow('cam3', grayscale_cam3)
                # # all_g = grayscale_cam3 * grayscale_cam2 * grayscale_cam1 * grayscale_cam0
                # # all_g = (all_g - all_g.min()) / (all_g.max() - all_g.min())
                # # cv2.imshow('all', all_g)
                # # cv2.imshow('mean', (grayscale_cam3 + grayscale_cam2 + grayscale_cam1 + grayscale_cam0) / 4)
                # grayscale_cam0 = (grayscale_cam0 * 255).astype(np.uint8)
                # ret, grayscale_cam0 = cv2.threshold(grayscale_cam0, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # grayscale_cam0 = torch.tensor(grayscale_cam0, device=device)
                # TP2 = torch.where(mask > 0, 1, 0) * torch.where(grayscale_cam0 > 0, 1, 0)
                # FP2 = torch.where(grayscale_cam0 > 0, 1, 0) * torch.where(mask > 0, 0, 1)
                # FN2 = torch.where(mask > 0, 1, 0) * torch.where(grayscale_cam0 > 0, 0, 1)
                #
                # m2 = torch.stack([FN2, TP2, FP2])
                # m2 = m2.permute(1, 2, 0).cpu().numpy().astype(np.float)
                #
                # gt_mask2 = deepcopy(brain_window)
                # gt_mask2[FN2 > 0] = 1
                #
                # predicted_mask2 = deepcopy(brain_window)
                # predicted_mask2[FP2 > 0] = 1
                #
                # correct_pred2 = deepcopy(brain_window)
                # correct_pred2[TP2 > 0] = 1
                #
                # segmentation_image2 = torch.stack([gt_mask2, correct_pred2, predicted_mask2])
                # segmentation_image2 = segmentation_image2.permute(1, 2, 0).cpu().numpy()
                # cv2.imshow('segmentation2', segmentation_image2)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\segmentation2-combined.bmp',
                #             255 * segmentation_image2)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\pmask2-combined.bmp', 255 * m2)
                # cv2.imshow('mask2', m2)

                cv2.waitKey()

        dices.append(dice)
        ious.append(iou)
        # plt.hist(dices)
        # plt.show()
    mean_dice = np.nanmean(dices)
    mean_iou = np.nanmean(ious)

    confusion_matrix.compute_confusion_matrix()
    return mean_dice, mean_iou, confusion_matrix


def segmentation_3d(model: torch.nn.Module, dataset: PhysioNetICHDataset, threshold, device='cpu'):
    model.to(device)
    model.eval()

    target_layers = [model.layers[-1].blocks[0]]

    attentions = {}
    for ln, layer in enumerate(model.layers):
        for bn, block in enumerate(layer.blocks):
            block.attn.softmax.register_forward_hook(get_attentions(attentions, f'{ln}_{bn}'))

    pbar = tqdm(dataset)
    pbar.set_description('segmentation')

    dices = []
    ious = []
    scans_num_slices = dataset.scans_num_slices
    mask_3d = []
    pred_mask_3d = []
    remaining_slices = scans_num_slices.pop(0)
    for sample, mask, label in pbar:
        if remaining_slices == 0:
            p_mask = torch.stack(pred_mask_3d)
            gt_mask = torch.stack(mask_3d)

            dice = dice_metric(p_mask, gt_mask).item()
            iou = intersection_over_union(p_mask, gt_mask).item()

            dices.append(dice)
            ious.append(iou)

            remaining_slices = scans_num_slices.pop(0)
            mask_3d = []
            pred_mask_3d = []

        sample, mask = sample.to(device), mask.to(device)
        sample = sample.unsqueeze(0)

        grad_cam = cam(sample)
        with torch.no_grad():
            attentions.clear()
            output = model(sample)
            if not torch.round(torch.sigmoid(output)).any():
                pred_mask = torch.zeros(sample.size(-2), sample.size(-1), device=device)
            else:
                pred_mask = get_attention_mask(attentions, threshold, grad_cam, device=device)

            mask_3d.append(mask)
            pred_mask_3d.append(pred_mask)

            remaining_slices -= 1

    print('dice: min={}, max={}, median={} mean={} std={}\n IoU: min={}, max={}, median={} mean={} std={}'.format(
        min(dices), max(dices),
        statistics.median(dices), statistics.mean(dices),
        statistics.stdev(dices), min(ious), max(ious),
        statistics.median(ious), statistics.mean(ious),
        statistics.stdev(ious)))