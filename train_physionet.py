import os
import argparse
import json

import monai.networks.nets
import numpy as np
import statistics
import monai
import torch
import pickle
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from collections import Counter

from helpers.preprocessing import Augmentation
from helpers.dataset import PhysioNetICHDataset, physio_collate_image_mask, physionet_cross_validation_split
from models.unet import UNet
from helpers.trainer import train
from helpers.utils import str_to_bool, DiceCELoss, EarlyStopping


def main():
    """
    Run this method to train the UNet model for Fully Supervised Hemorrhage segmentation.
    Training parameters is defined in the config file: train_physionet_config.json.
    """
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="configs/train_physionet_config.json")
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

    model_name = config_dict["model_name"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]
    embed_dims = list(config_dict["embed_dims"])

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation(with_mask=True)

    folds = 5
    physionet_cross_validation_split(data_path, extra_path, k=folds, override=False)
    # creating the physionet dataset.
    # brain and bone window parameters are derived from the dataset paper. Subdural window params is adapted based on RSNA one, and these two ds differences.
    ds = PhysioNetICHDataset(data_path, windows=[(80, 340), (700, 3200)])
    all_indices = np.arange(0, len(ds.labels))
    for cf in range(0, folds):
        # get the indices for train, validation and test sets.
        with open(os.path.join(extra_path, "folds_division", f"fold{cf}.pt"), "rb") as test_indices_file:
            test_indices = pickle.load(test_indices_file)
        train_valid_indices = [x for x in all_indices if x not in test_indices]
        # train/validation division is only based on whether hemorrhages exist, not the subtypes
        train_indices, valid_indices = train_test_split(train_valid_indices, stratify=ds.labels[train_valid_indices, -1], test_size=validation_ratio, random_state=42)

        if model_name == "UNet":
            model = UNet(in_ch=in_ch, num_classes=num_classes, embed_dims=embed_dims)
        else:
            model = monai.networks.nets.SwinUNETR(img_size=(384, 384), in_channels=in_ch, out_channels=num_classes, spatial_dims=2,
                                                  depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), feature_size=12)
        # we use a combo loss to overcome to problem of imbalanced foreground/background pixels.
        loss_fn = DiceCELoss()
        checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__

        if do_sampling:
            _labels = ds.labels
            # getting only the label that represents "ANY" hemorrhage
            _labels = _labels[:, -1]
            labels_counts = Counter(_labels[train_indices])
            target_list = torch.LongTensor(_labels)
            # weight the sampler inversely proportional to the labels frequency
            weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
            class_weights = weights[target_list]
            # zero-out the weight of test-set and validation-set in order for the sample to only get data from train-set
            class_weights[test_indices] = 0
            class_weights[valid_indices] = 0
            # because the dataset is small, we make sampling with-replacement and the size equal to the dataset size,
            # so at each epoch, positive samples are seen multiple times, and as a result, some negative samples remain unseen.
            # However, after several epochs, all samples is likely to be seen. This technique has shown to be the most effective one.
            # on the other hand, for RSNA sampler, we make it "without" replacement, and the data size double than the minority class,
            # as it makes it faster with a little performance loss.
            train_sampler = WeightedRandomSampler(class_weights, len(train_indices), replacement=True)
            train_loader = DataLoader(ds, batch_size=batch_size, collate_fn=physio_collate_image_mask, num_workers=num_workers, sampler=train_sampler)
        else:
            # if sampling is not requested, we do normal batching.
            train_ds = Subset(ds, train_indices)
            train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=physio_collate_image_mask, num_workers=num_workers)
        valid_ds = Subset(ds, valid_indices)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, collate_fn=physio_collate_image_mask)

        opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        early_stopping = EarlyStopping(model, 3, os.path.join(extra_path, "weights", f"{checkpoint_name}-fold{cf}.pth"))
        print(f"training model: {checkpoint_name}")

        train(early_stopping, 1, model, opt, loss_fn, train_loader, valid_loader, augmentation=augmentation,
              result_plot_path=os.path.join(extra_path, 'plots', f'train_model_{checkpoint_name}-fold{cf}'))
        del model, opt, early_stopping, loss_fn
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
