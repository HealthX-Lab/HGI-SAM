import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import json
import numpy as np
from helpers.dataset import rsna_train_valid_split, RSNAICHDataset, rsna_collate_binary_label
from helpers.preprocessing import get_transform, Augmentation
from helpers.preprocessing import *
from helpers.utils import *
from helpers.trainer import train_one_epoch, train_one_epoch_refine_segmentation
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from models.swin_weak import SwinWeak
import torch
from collections import Counter
import cv2
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/refine_rsna_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    do_augmentation = str_to_bool(config_dict["do_augmentation"])
    do_sampling = str_to_bool(config_dict["do_sampling"])
    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    t_x, t_y, v_x, v_y = rsna_train_valid_split(data_path, override=False)
    t_x_pos, t_y_pos, v_x_pos, v_y_pos = [], [], [], []
    for x, y in zip(t_x, t_y):
        if y[-1] == 1:
            t_x_pos.append(x)
            t_y_pos.append(y)
    for x, y in zip(v_x, v_y):
        if y[-1] == 1:
            v_x_pos.append(x)
            v_y_pos.append(y)

    windows = [(80, 200), (600, 2800)]
    transform = get_transform(img_size)

    train_ds = RSNAICHDataset(data_path, t_x_pos[:100], t_y_pos[:100], windows=windows, transform=transform)
    validation_ds = RSNAICHDataset(data_path, v_x_pos[:10], v_y_pos[:10], windows=windows, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label)
    valid_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label)

    model = SwinWeak(in_ch, num_classes)
    load_model(model.swin, r"C:\rsna-ich\Good weights\backup\Focal-100-checkpoint-0.pt")
    loss_fn = DiceCELoss()
    checkpoint_name = model.__class__.__name__ + "_refined"
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: ", checkpoint_name, " num-params:", num_params)

    opt = AdamW(model.refinement_unet.parameters(), lr=lr, weight_decay=1e-6)
    early_stopping = EarlyStopping(model, 3, os.path.join(extra_path, f"weights/{checkpoint_name}.pt"))
    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epochs <= epoch_number:
        _metrics = train_one_epoch_refine_segmentation(model, opt, loss_fn, train_loader, valid_loader)
        val_loss = _metrics['valid_cfm'].get_mean_loss()

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)
        visualize_losses(train_losses, valid_losses)

        early_stopping(val_loss)
        epoch_number += 1


if __name__ == '__main__':
    main()
