import nibabel
import os
import cv2
import numpy as np
from tqdm import tqdm
from dataset import *
from torchvision import transforms
from utils import *
import torch.nn as nn
from train import *
import torch
from preprocessing import *
from torch.utils.data import DataLoader, Subset
import timm
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from models.unet import *
from models.swin_unetr import *
from models.swin_weak import *
from models.swin_weak import *
import torch.optim as optim
from utils import *
import statistics


def main():
    # train_and_test_physionet()
    train_rsna()


def train_rsna():
    root_dir = r'C:\rsna-ich'

    t_x, t_y, v_x, v_y = rsna_2d_train_validation_split(root_dir)
    train_ds = RSNAICHDataset2D(root_dir, t_x, t_y, transform=get_transform(384))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True, collate_fn=rsna_collate_binary_label)
    validation_ds = RSNAICHDataset2D(root_dir, v_x, v_y, transform=get_transform(384))
    valid_loader = DataLoader(validation_ds, batch_size=16, collate_fn=rsna_collate_binary_label)

    model = SwinWeak(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = FocalLoss()
    early_stopping = EarlyStopping(model, 3, r'weights\swin-weak.pth')
    epoch = 1
    while not early_stopping.early_stop:
        m = train_one_epoch(model, optimizer, loss_fn, train_loader, valid_loader)
        early_stopping(m['valid_cfm'].get_mean_loss())
        print("epoch:", epoch)
        print("train-loss=", m['train_cfm'].get_mean_loss(), " valid-loss=", m['valid_cfm'].get_mean_loss())
        print("train-acc=", m['train_cfm'].get_accuracy(), " valid-acc=", m['valid_cfm'].get_accuracy())
        print("train-F1=", m['train_cfm'].get_f1_score(), " valid-F1=", m['valid_cfm'].get_f1_score())
        print()


def train_and_test_physionet():
    k = 10
    device = 'cuda'
    checkpoint_name = r'weights\swin-unetr'
    transform = get_transform(384)
    augmentation = Augmentation(device)

    ds = PhysioNetICHDataset2D(r'C:\physio-ich', transform=transform)

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

        # model = UNet(1, 1)
        # model = SwinUNetR(1, 1)
        model = SwinWeak(1, 1)
        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # # loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        # # loss_fn = DiceBCELoss()
        # # loss_fn = DiceLoss()
        # # loss_fn = FocalDiceBCELoss()
        # loss_fn = FocalDiceLoss()
        # early_stopping = EarlyStopping(model, 3, f'{checkpoint_name}-fold{cf}.pth')
        # while not early_stopping.early_stop:
        #     m = train_one_epoch_segmentation(model, optimizer, loss_fn, train_loader, valid_loader, augmentation=augmentation, device=device)
        #     print('\nvalid dice:', m['valid_cfm'].get_mean_dice(), ' IoU:', m['valid_cfm'].get_mean_iou(), ' Hausdorff:', m['valid_cfm'].get_mean_hausdorff_distance())
        #     early_stopping(m['valid_cfm'].get_mean_loss())
        #
        # load_model(model, f'{checkpoint_name}-fold{cf}.pth')
        test_cfm = test_physionet(model, test_loader, True, 0.01, device)

        test_cfm_matrices.append(test_cfm)
        print(f'fold {cf+1} dice:', test_cfm.get_mean_dice(), ' iou:', test_cfm.get_mean_iou(), ' hausdorff:', test_cfm.get_mean_hausdorff_distance())

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
                pred = model.segmentation(image)
            else:
                pred = model(image)
            pred_mask = torch.sigmoid(pred).squeeze(1)
            pred_mask = binarization_simple_thresholding(pred_mask, threshold)
            test_cfm.add_dice(dice_metric(pred_mask, label))
            test_cfm.add_iou(intersection_over_union(pred_mask, label))
            test_cfm.add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    return test_cfm


if __name__ == '__main__':
    main()
