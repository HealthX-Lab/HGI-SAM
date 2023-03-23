import torch.optim
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
from monai.transforms.post.array import one_hot


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda'):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)

        pred = model(sample)
        pred = F.softmax(pred, dim=1)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_prediction(torch.argmax(pred, dim=1), label)

    _metrics["train_cfm"].compute_confusion_matrix()

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            pred = F.softmax(pred, dim=1)
            loss = loss_fn(pred, label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_prediction(torch.argmax(pred, dim=1), label)

        _metrics["valid_cfm"].compute_confusion_matrix()

    return _metrics


def train_one_epoch_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        if not label.any():
            continue
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)
        if augmentation:
            sample, label = augmentation(sample, label)

        pred = model(sample)
        loss = loss_fn(pred.squeeze(1), label)

        loss.backward()
        optimizer.step()
        pred_mask = torch.round(torch.sigmoid(pred.squeeze(1)))

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))
        _metrics["train_cfm"].add_dice(dice_metric(pred_mask, label))
        _metrics["train_cfm"].add_iou(intersection_over_union(pred_mask, label))
        _metrics["train_cfm"].add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            if not label.any():
                continue
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            loss = loss_fn(pred.squeeze(1), label)
            pred_mask = torch.round(torch.sigmoid(pred.squeeze(1)))

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))
            _metrics["valid_cfm"].add_dice(dice_metric(pred_mask, label))
            _metrics["valid_cfm"].add_iou(intersection_over_union(pred_mask, label))
            _metrics["valid_cfm"].add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    return _metrics


def train_one_epoch_refine_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample = sample.to(device)

        with torch.no_grad():
            p, p_mask = model.attentional_segmentation(sample)
            p_mask = binarization_simple_thresholding(p_mask, 0.07)

        if augmentation:
            sample, p_mask = augmentation(sample, p_mask)

        p_mask = one_hot(p_mask.unsqueeze(1), 2, dim=1)
        pred = model.refinement_segmentation(sample)
        loss = loss_fn(pred, p_mask)

        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample = sample.to(device)

            p, p_mask = model.attentional_segmentation(sample)
            p_mask = binarization_simple_thresholding(p_mask, 0.07)

            p_mask = one_hot(p_mask.unsqueeze(1), 2, dim=1)
            pred = model.refinement_segmentation(sample)
            loss = loss_fn(pred, p_mask)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))

    return _metrics