import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import json
from helpers.preprocessing import Augmentation
from helpers.utils import EarlyStopping, ConfusionMatrix, dice_metric, hausdorff_distance, intersection_over_union, binarization_otsu, binarization_simple_thresholding, load_model
from helpers.dataset import PhysioNetICHDataset, physio_collate_image_mask, physionet_cross_validation_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from models.unet import UNet
from helpers.trainer import train_one_epoch_segmentation
from helpers.preprocessing import get_transform
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
import torch
import torch.nn as nn
import pickle
from helpers.utils import str_to_bool, visualize_losses, DiceCELoss
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_physionet_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    do_augmentation = str_to_bool(config_dict["do_augmentation"])
    do_sampling = str_to_bool(config_dict["do_sampling"])
    validation_ratio = config_dict["validation_ratio"]
    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]
    embed_dims = list(config_dict["embed_dims"])

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation(with_mask=True)

    k = 5
    # physionet_cross_validation_split()

    ds = PhysioNetICHDataset(data_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))
    all_indices = np.arange(0, len(ds.labels))
    for cf in range(4, k):
        with open(os.path.join(extra_path, "folds_division", f"fold{cf}.pt"), "rb") as test_indices_file:
            test_indices = pickle.load(test_indices_file)
        train_valid_indices = [x for x in all_indices if x not in test_indices]
        train_indices, valid_indices = train_test_split(train_valid_indices, stratify=ds.labels[train_valid_indices, -1], test_size=validation_ratio, random_state=42)

        model = UNet(in_ch=in_ch, num_classes=num_classes, embed_dims=embed_dims)
        loss_fn = DiceCELoss()
        checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__
        checkpoint_path = os.path.join(extra_path, "weights", checkpoint_name)

        if do_sampling:
            _labels = ds.labels
            _labels = _labels[:, -1]
            labels_counts = Counter(_labels[train_indices])
            target_list = torch.LongTensor(_labels)
            weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
            class_weights = weights[target_list]
            class_weights[test_indices] = 0
            class_weights[valid_indices] = 0
            train_sampler = WeightedRandomSampler(class_weights, len(train_indices), replacement=True)
            train_loader = DataLoader(ds, batch_size=batch_size, collate_fn=physio_collate_image_mask, num_workers=num_workers, sampler=train_sampler)
        else:
            train_ds = Subset(ds, train_indices)
            train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=physio_collate_image_mask, num_workers=num_workers, shuffle=True)
        valid_ds = Subset(ds, valid_indices)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, collate_fn=physio_collate_image_mask)

        train_physionet(model, lr, epochs, loss_fn, train_loader, valid_loader, checkpoint_path, cf, augmentation=augmentation)
        del model
        torch.cuda.empty_cache()


def train_physionet(model: nn.Module, lr, epochs, loss_fn, train_loader, valid_loader, checkpoint_path, cf, device='cuda', augmentation=None):
    opt = AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(model, 3, f'{checkpoint_path}-fold{cf}')

    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epochs <= epoch_number:
        _metrics = train_one_epoch_segmentation(model, opt, loss_fn, train_loader, valid_loader, augmentation=augmentation)
        val_loss = _metrics['valid_cfm'].get_mean_loss()

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)
        visualize_losses(train_losses, valid_losses)

        print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n")
        early_stopping(val_loss, epoch_number)
        epoch_number += 1


if __name__ == '__main__':
    main()


def train_and_test_physionet(physio_path, model, loss_fn):
    k = 5
    device = 'cuda'
    checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__

    ds = PhysioNetICHDataset(physio_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))

    indices = np.arange(0, len(ds.labels))
    encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in ds.labels])
    skf = StratifiedKFold(k)
    test_cfm_matrices = []
    for cf, (train_valid_indices, test_indices) in enumerate(skf.split(indices, encoded_labels)):  # dividing intro train/test based on all subtypes
        # dividing into train/valid based on any hemorrhage
            train_indices, valid_indices = train_test_split(train_valid_indices, stratify=ds.labels[train_valid_indices, -1], test_size=1. / (k - 1), random_state=42)

    train_ds = Subset(ds, train_indices)
    valid_ds = Subset(ds, valid_indices)
    test_ds = Subset(ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=physio_collate_image_mask)
    valid_loader = DataLoader(valid_ds, batch_size=1, collate_fn=physio_collate_image_mask)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=physio_collate_image_mask)

    train_physionet(model, loss_fn, train_loader, valid_loader, checkpoint_name, cf, device)
    load_model(model, f'{checkpoint_name}-fold{cf}.pth')
    test_cfm = test_physionet(model, test_loader, False, 0.5, device)

    test_cfm_matrices.append(test_cfm)
    print(f'fold {cf + 1} dice:', test_cfm.get_mean_dice(), ' iou:', test_cfm.get_mean_iou(), ' hausdorff:', test_cfm.get_mean_hausdorff_distance())

    dices = []
    IoUs = []
    hausdorff_distances = []
    for i in range(k):
        dices.append(test_cfm_matrices[i].get_mean_dice().item())
        IoUs.append(test_cfm_matrices[i].get_mean_iou().item())
        hausdorff_distances.append(test_cfm_matrices[i].get_mean_hausdorff_distance().item())

    print('dice: ', statistics.mean(dices), ' +/- ', statistics.stdev(dices))
    print('IoU: ', statistics.mean(IoUs), ' +/- ', statistics.stdev(IoUs))
    print('hausdorff: ', statistics.mean(hausdorff_distances), ' +/- ', statistics.stdev(hausdorff_distances))


def test_physionet(model, test_loader, weak_model=False, threshold=0.5, device="cuda"):
    model.to(device)
    test_cfm = ConfusionMatrix()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            if not label.any():
                continue
            if weak_model:
                pred, pred_mask = model.attentional_segmentation(image)
            else:
                pred_mask = model(image)
                pred_mask = torch.sigmoid(pred_mask).squeeze(1)

            pred_mask = binarization_simple_thresholding(pred_mask, threshold)
            test_cfm.add_dice(dice_metric(pred_mask, label))
            test_cfm.add_iou(intersection_over_union(pred_mask, label))
            test_cfm.add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    return test_cfm
