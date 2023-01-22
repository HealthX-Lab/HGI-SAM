import nibabel
import os
import cv2
from tqdm import tqdm
from dataset import *
from models.swin_transformer_v2 import *
from torchvision import transforms
from utils import *
import torch.nn as nn
from train import *
import torch
from preprocessing import *
from torch.utils.data import DataLoader


def main():
    src = r'D:\Datasets\rsna-ich-nifty'
    train_files, validation_files = rsna_3d_train_validation_split(src)
    transform = transforms.Compose([
        transforms.Resize(int(224 * 1.2)),
        transforms.CenterCrop(224),
    ])
    train_ds = RSNAICHDataset3D(src, train_files[756:1000], windows=[(40, 80), (40, 80), (40, 80)], transform=transform)
    train_dl = DataLoader(train_ds, batch_size=4, collate_fn=rsna_collate_fn)
    validation_ds = RSNAICHDataset3D(src, validation_files[:100], windows=[(40, 80), (40, 80), (40, 80)], transform=transform)
    validation_dl = DataLoader(validation_ds, batch_size=4, collate_fn=rsna_collate_fn)
    model = SwinTransformerV2(num_classes=1)
    opt = torch.optim.Adam(model.parameters())
    for i in range(10):
        m = train_one_epoch(model, opt, FocalBCELoss(), train_dl, validation_dl)
        print(m["train_loss"], m["valid_loss"])
        print(m["train_cm"].get_f1_score(), m["valid_cm"].get_f1_score())
        print(m["train_cm"].get_accuracy(), m["valid_cm"].get_accuracy())


if __name__ == '__main__':
    main()
