import torch.optim
from utils import *
from tqdm import tqdm
from preprocessing import Augmentation


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda'):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}
    augmentation = Augmentation()

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample, label = augmentation(sample.to(device)), label.to(device)

        pred = model(sample).view(-1)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_prediction(torch.sigmoid(pred), label)

    _metrics["train_cfm"].compute_confusion_matrix()

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader))
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample).view(-1)
            loss = loss_fn(pred, label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_prediction(torch.sigmoid(pred), label)

        _metrics["valid_cfm"].compute_confusion_matrix()

    return _metrics


def train_one_epoch_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        # if not label.any():
        #     continue
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)
        if augmentation:
            sample, label = augmentation(sample, label)

        pred = model(sample)
        loss = loss_fn(pred.squeeze(1), label)

        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))
        _metrics["train_cfm"].add_dice(dice_metric(torch.round(torch.sigmoid(pred.squeeze(1))), label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader))
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            # if not label.any():
            #     continue
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            loss = loss_fn(pred.squeeze(1), label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))
            _metrics["valid_cfm"].add_dice(dice_metric(torch.round(torch.sigmoid(pred.squeeze(1))), label))

    return _metrics
