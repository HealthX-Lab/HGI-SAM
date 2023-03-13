import argparse
import json
from utils.preprocessing import Augmentation
from utils.utils import EarlyStopping, DiceBCELoss, ConfusionMatrix, dice_metric, hausdorff_distance, intersection_over_union, binarization_otsu, binarization_simple_thresholding, load_model
from utils.dataset import PhysioNetICHDataset2D, physio_collate_image_mask
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from models.unet import UNet
from utils.train import train_one_epoch_segmentation
from utils.preprocessing import get_transform
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_physionet_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    data_path = config_dict["data_path"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]


def train_physionet(model: nn.Module, loss_fn, train_loader, valid_loader, checkpoint_name, cf, device='cuda'):
    augmentation = Augmentation(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(model, 3, f'{checkpoint_name}-fold{cf}.pth')
    while not early_stopping.early_stop:
        m = train_one_epoch_segmentation(model, optimizer, loss_fn, train_loader, valid_loader, augmentation=augmentation, device=device)
        print('\nvalid dice:', m['valid_cfm'].get_mean_dice(), ' IoU:', m['valid_cfm'].get_mean_iou(), ' Hausdorff:', m['valid_cfm'].get_mean_hausdorff_distance())
        early_stopping(m['valid_cfm'].get_mean_loss())


if __name__ == '__main__':
    main()


def train_and_test_physionet(physio_path, model, loss_fn):
    k = 10
    device = 'cuda'
    checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__

    ds = PhysioNetICHDataset2D(physio_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))

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
                pred, pred_mask = model.segmentation(image)
            else:
                pred_mask = model(image)
                pred_mask = torch.sigmoid(pred_mask).squeeze(1)

            pred_mask = binarization_simple_thresholding(pred_mask, threshold)
            test_cfm.add_dice(dice_metric(pred_mask, label))
            test_cfm.add_iou(intersection_over_union(pred_mask, label))
            test_cfm.add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    return test_cfm
