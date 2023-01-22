import nibabel
import os
import cv2
from tqdm import tqdm
from dataset import *
from torchvision import transforms
from utils import *
import torch.nn as nn
from train import *
import torch
from preprocessing import *
from torch.utils.data import DataLoader
import timm


def main():
    print(timm.list_models('*swin*'))
    src = r'C:\rsna-ich-nifty'
    train_files, validation_files = rsna_3d_train_validation_split(src)
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
    ])
    train_ds = RSNAICHDataset3D(src, train_files[1000:1100], windows=[(40, 80)], transform=transform)
    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=rsna_collate_fn, shuffle=True)
    validation_ds = RSNAICHDataset3D(src, validation_files[500:510], windows=[(40, 80)], transform=transform)
    validation_dl = DataLoader(validation_ds, batch_size=1, collate_fn=rsna_collate_fn)
    model = timm.create_model('swin_base_patch4_window7_224', in_chans=1, num_classes=1)
    opt = torch.optim.Adam(model.parameters())
    for i in range(10):
        m = train_one_epoch(model, opt, FocalBCELoss(), train_dl, validation_dl)
        print(m["train_loss"], m["valid_loss"])
        print(m["train_cm"].get_f1_score(), m["valid_cm"].get_f1_score())
        print(m["train_cm"].get_accuracy(), m["valid_cm"].get_accuracy())


if __name__ == '__main__':
    main()
