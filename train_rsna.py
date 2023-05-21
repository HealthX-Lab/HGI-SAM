import os
import argparse
import json
from helpers.dataset import rsna_train_valid_split, RSNAICHDataset, rsna_collate_binary_label
from helpers.preprocessing import *
from helpers.utils import *
from helpers.trainer import train
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from models.swin_weak import SwinWeak
import torch
from collections import Counter
import torch.nn as nn


def main():
    """
    Run this method to train the SwinWeak model for binary ICH detection task.
    Training parameters is defined in the config file: train_rsna_config.json.
    """
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="configs/train_rsna_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    do_augmentation = str_to_bool(config_dict["do_augmentation"])
    do_sampling = str_to_bool(config_dict["do_sampling"])
    do_finetune = str_to_bool(config_dict["do_finetune"])
    validation_ratio = config_dict["validation_ratio"]
    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    t_x, t_y, v_x, v_y = rsna_train_valid_split(data_path, extra_path, validation_size=validation_ratio, override=False)

    windows = [(80, 200), (600, 2800)]  # window-center and window-width for windowing CT HF values into subdural and bone windows respectively.

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation(with_mask=False)

    train_ds = RSNAICHDataset(data_path, t_x, t_y, windows=windows, augmentation=augmentation)
    validation_ds = RSNAICHDataset(data_path, v_x, v_y, windows=windows)

    train_sampler = None
    if do_sampling:
        # initializing a sampler that ensures number of positive and negative samples in a batch is almost the same
        _labels = train_ds.labels
        _labels = _labels[:, -1]  # taking the "any" label to do sampling which corresponds to any hemorrhage
        labels_counts = Counter(_labels)
        target_list = torch.LongTensor(_labels)
        weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
        class_weights = weights[target_list]
        train_sampler = WeightedRandomSampler(class_weights, 2 * labels_counts[1], replacement=False)  # the size of which we do sampling from is 2 x minority-class

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label, sampler=train_sampler)
    valid_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label)

    model = SwinWeak(in_ch, num_classes, pretrained=False if do_finetune else True)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_name = model.__class__.__name__ + "_" + loss_fn.__class__.__name__
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: ", checkpoint_name, " num-params:", num_params)

    if do_finetune:
        mlcn_binary_model = SwinWeak(in_ch, 1, pretrained=False)
        load_model(mlcn_binary_model, os.path.join(extra_path, 'weights/backup/SwinWeak_FocalBinaryCrossEntropyLoss-binary.pth'))
        model.swin.load_state_dict(mlcn_binary_model.swin.state_dict())
        for param in model.swin.parameters():
            param.requires_grad = False
        opt = AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-6)
        early_stopping = EarlyStopping(model, 3, os.path.join(extra_path, f"weights/finetune_{checkpoint_name}.pth"))
        print(f"training head: {checkpoint_name}")
        train(early_stopping, -1, model, opt, loss_fn, train_loader, valid_loader,
              result_plot_path=os.path.join(extra_path, 'plots', f'train_head_{checkpoint_name}'))
        load_model(model, os.path.join(extra_path, f"weights/finetune_{checkpoint_name}.pth"))

    for param in model.parameters():
        param.requires_grad = True
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    early_stopping = EarlyStopping(model, 3, os.path.join(extra_path, "weights", f"{checkpoint_name}.pth"))
    print(f"training model: {checkpoint_name}")
    train(early_stopping, epochs, model, opt, loss_fn, train_loader, valid_loader,
          result_plot_path=os.path.join(extra_path, 'plots', f'train_model_{checkpoint_name}'))


if __name__ == '__main__':
    main()