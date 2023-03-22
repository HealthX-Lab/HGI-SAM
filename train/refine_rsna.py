import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import json
import numpy as np
from utils.dataset import rsna_train_valid_split, RSNAICHDataset, rsna_collate_binary_label
from utils.preprocessing import get_transform, Augmentation
from utils.preprocessing import *
from utils.utils import *
from utils.trainer import train_one_epoch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from models.swin_weak import SwinWeak
import torch
from collections import Counter
import cv2
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_rsna_config.json")
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
    windows = [(80, 200), (600, 2800)]
    transform = get_transform(img_size)

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation(with_mask=False)

    train_ds = RSNAICHDataset(data_path, t_x, t_y, windows=windows, transform=transform, augmentation=augmentation)
    validation_ds = RSNAICHDataset(data_path, v_x, v_y, windows=windows, transform=transform)

    train_sampler = None
    if do_sampling:
        _labels = train_ds.labels
        _labels = _labels[:, -1]
        labels_counts = Counter(_labels)
        target_list = torch.LongTensor(_labels)
        weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
        class_weights = weights[target_list]
        train_sampler = WeightedRandomSampler(class_weights, 2 * labels_counts[1], replacement=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label, sampler=train_sampler)
    valid_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label)

    # for i, (x, y) in enumerate(train_loader):
    #     z = x[0].permute(1, 2, 0)
    #     cv2.imshow('img', z.numpy())
    #     cv2.waitKey()
    #     if i > 100:
    #         return

    model = SwinWeak(in_ch, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_name = model.__class__.__name__ + "_" + loss_fn.__class__.__name__
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: ", checkpoint_name, " num-params:", num_params)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    early_stopping = EarlyStopping(model, 3, os.path.join(extra_path, f"weights/{checkpoint_name}.pt"))
    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epochs <= epoch_number:
        _metrics = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader)
        val_loss = _metrics['valid_cfm'].get_mean_loss()

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)
        visualize_losses(train_losses, valid_losses)

        print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n"
              f"train-acc:{_metrics['train_cfm'].get_accuracy()}, valid-acc:{_metrics['valid_cfm'].get_accuracy()}\n"
              f"train-F1:{_metrics['train_cfm'].get_f1_score()}, valid-F1:{_metrics['valid_cfm'].get_f1_score()}")
        early_stopping(val_loss)
        epoch_number += 1


if __name__ == '__main__':
    main()
