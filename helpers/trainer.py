import torch.optim
from helpers.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from models.unet import UNet
from models.swin_weak import SwinWeak
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

    return _metrics


def train(early_stopping: EarlyStopping, epochs, model, opt, loss_fn, train_loader, valid_loader, result_plot_path, augmentation=None):
    """
    A method to do the training until it early stops as a result of not seeing a reduction in validation loss

    :param early_stopping: an object that controls early stopping
    :param epochs: for how many epochs to model should be trained; -1 means it should be trained until it early stops
    :param model: the model to train (Swin-Weak or UNet)
    :param opt: optimizer object
    :param loss_fn: loss function object
    :param train_loader: data loader for train set
    :param valid_loader: data loader for test set
    :param result_plot_path: the path to save the train/validation loss plots
    :param augmentation: only used when trained in segmentation mode. Because in PhysioNet dataset we read the whole dataset first, we have to do the augmentation while training.
    """
    epochs = np.inf if epochs == -1 else epochs
    _type = "classification" if isinstance(model, SwinWeak) else "segmentation"
    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epoch_number <= epochs:
        if _type == "classification":
            _metrics = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader)
            val_loss = _metrics['valid_cfm'].get_mean_loss()
            _metrics["train_cfm"].compute_confusion_matrix()
            _metrics["valid_cfm"].compute_confusion_matrix()
            print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n"
                  f"train-acc:{_metrics['train_cfm'].get_accuracy()}, valid-acc:{_metrics['valid_cfm'].get_accuracy()}\n"
                  f"train-F1:{_metrics['train_cfm'].get_f1_score()}, valid-F1:{_metrics['valid_cfm'].get_f1_score()}")

        else:
            _metrics = train_one_epoch_segmentation(model, opt, loss_fn, train_loader, valid_loader, augmentation=augmentation)
            val_loss = _metrics['valid_cfm'].get_mean_loss()
            print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n")

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)

        early_stopping(val_loss)
        epoch_number += 1
    visualize_losses(train_losses, valid_losses, result_plot_path)


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
