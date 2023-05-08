import torch.optim
from helpers.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
from monai.transforms.post.array import one_hot


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda'):
    """
    helper method to configs a model for one epoch
    :param model: model to configs
    :param optimizer: optimizer to use (ADAMW)
    :param loss_fn: loss function
    :param train_loader: data loader for configs set
    :param valid_loader: data loader for validation set
    :param device: whether to configs the model on cuda or cpu
    :return: evaluation metrics of both configs and validation sets
    """
    model.to(device)
    model.train()
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
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
        _metrics["train_cfm"].add_number_of_samples(len(label))

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
            _metrics["valid_cfm"].add_number_of_samples(len(label))

        _metrics["valid_cfm"].compute_confusion_matrix()

    return _metrics


def train_one_epoch_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)
        if augmentation is not None:
            for b in range(len(label)):
                sample[b], label[b] = augmentation(sample[b], label[b])

        pred = model(sample)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            loss = loss_fn(pred, label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))

    return _metrics
