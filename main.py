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
from collections import Counter


def main():
    # print(timm.list_models('*swin*'))
    # src = r'C:\rsna-ich-nifty'
    # train_files, validation_files = rsna_3d_train_validation_split(src, override=False)
    # img_size = 224
    # transform = transforms.Compose([
    #     transforms.Resize(int(img_size * 1.2)),
    #     transforms.CenterCrop(img_size),
    # ])
    # train_ds = RSNAICHDataset3D(src, train_files[14125:], windows=[(40, 80)], transform=transform)
    # train_dl = DataLoader(train_ds, batch_size=1, collate_fn=rsna_collate_fn, shuffle=False)
    # validation_ds = RSNAICHDataset3D(src, validation_files, windows=[(40, 80)], transform=transform)
    # validation_dl = DataLoader(validation_ds, batch_size=1, collate_fn=rsna_collate_fn)
    # model = timm.create_model('swin_base_patch4_window7_224', in_chans=1, num_classes=1)
    # opt = torch.optim.Adam(model.parameters())
    # early_stopping = EarlyStopping(model, patience=3, path_to_save='model.pth')
    # cfm = None
    # while not early_stopping.early_stop:
    #     cfm = train_one_epoch(model, opt, FocalBCELoss(), train_dl, validation_dl)
    #     early_stopping(cfm["valid_cfm"].get_mean_loss())
    # print('loss: ', cfm["train_cfm"].get_mean_loss(), cfm["valid_cfm"].get_mean_loss())
    # print('F1-score: ', cfm["train_cfm"].get_f1_score(), cfm["valid_cfm"].get_f1_score())
    # print('Accuracy: ', cfm["train_cfm"].get_accuracy(), cfm["valid_cfm"].get_accuracy())
    # print('Specificity: ', cfm["train_cfm"].get_specificity(), cfm["valid_cfm"].get_specificity())
    # print('Precision: ', cfm["train_cfm"].get_precision(), cfm["valid_cfm"].get_precision())
    # print('Recall: ', cfm["train_cfm"].get_recall_sensitivity(), cfm["valid_cfm"].get_recall_sensitivity())
    # print('AUC: ', cfm["train_cfm"].get_auc_score(), cfm["valid_cfm"].get_auc_score())
    # x, y, vx, vy = rsna_2d_train_validation_split(r'D:\Datasets\rsna-ich')
    # ds = RSNAICHDataset2D(r'D:\Datasets\rsna-ich', x, y, windows=[(80, 200, -1024, 1)])
    #
    # for a, b in ds:
    #     print(a.shape)
    #     cv2.imshow('a', a.numpy()[0, :, :])
    #     cv2.imshow('b', a.numpy()[1, :, :])
    #     print(b)
    #     cv2.waitKey()
    ds = PhysioNetICHDataset2D(r'C:\physio-ich')
    for i, m, l in ds:
        print(i.shape)
        cv2.imshow('i', i.numpy()[0, :, :])
        print(m.shape)
        cv2.imshow('m', m.numpy())
        print(l)
        cv2.waitKey()


if __name__ == '__main__':
    main()
