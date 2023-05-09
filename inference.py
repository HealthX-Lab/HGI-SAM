import os

import torch
import timm
from helpers.dataset import *
from helpers.preprocessing import get_transform
from models.swin_weak import SwinWeak
import cv2
from helpers.utils import *
import numpy as np
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, compute_meandice
from models.unet import UNet
from torch.utils.data import Subset
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from skimage import morphology
import matplotlib.pyplot as plt
# from segmentation_mask_overlay import overlay_masks
import cv2
import time
import argparse
import json


def main():
    """
    Run this method to test the trained models.
    model and their corresponding parameters is defined in the config file: inference_config.json
    """
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="configs/inference_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]
    models_weights = {"SwinSAM-binary": config_dict["SwinSAM-binary_path"],
                      "SwinSAM-multi": config_dict["SwinSAM-multi_path"],
                      "Swin-GradCAM": config_dict["Swin-GradCAM_path"],
                      "Swin-HGI-SAM": config_dict["Swin-HGI-SAM_path"],
                      "UNet": config_dict["UNet_path"]}
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ds = PhysioNetICHDataset(data_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))
    # dictionary holding segmentation results
    seg_results = {"dice": {"SwinSAM-binary": [], "SwinSAM-multi": [], "Swin-GradCAM": [], "Swin-HGI-SAM": [], "UNet": []},
                   "iou":  {"SwinSAM-binary": [], "SwinSAM-multi": [], "Swin-GradCAM": [], "Swin-HGI-SAM": [], "UNet": []}}

    # iterating over folds
    for fold_number in [0, 1, 2, 3, 4]:
        with open(os.path.join(extra_path, "folds_division", f"fold{fold_number}.pt"), 'rb') as test_indices_file:
            test_indices = pickle.load(test_indices_file)
        test_ds = Subset(ds, test_indices)  # used for evaluating the model
        val_indices = [x for x in range(0, len(ds.labels)) if x not in test_indices]
        val_ds = Subset(ds, val_indices)  # used in weakly supervised models only to find the optimum threshold for binarization

        models = {"SwinSAM-binary": SwinWeak(in_ch=3, num_classes=1),  # here, we use sigmoid + rounding to do binary classification
                  "SwinSAM-multi": SwinWeak(in_ch=3, num_classes=6),
                  # in following, we use softmax + argmax to do binary classification,
                  "Swin-GradCAM": SwinWeak(in_ch=3, num_classes=2),
                  "Swin-HGI-SAM": SwinWeak(in_ch=3, num_classes=2),
                  "UNet": UNet(in_ch=3, num_classes=2, embed_dims=[24, 48, 96, 192])}

        for model_name, model in models.items():
            # only for UNet model, we have to load the weight corresponding to the fold
            # thus, the path in the config file for UNet model is the path to the directory that contains the weights
            if model_name == "UNet":
                load_model(model, os.path.join(models_weights[model_name], f"UNet-DiceCELoss-fold{fold_number}.pt"))
            # for weak models that are trained on a separate dataset, we use the same weights to test different folds
            # thus, the path in the file for those models are the path to the corresponding file
            else:
                load_model(model, models_weights[model_name])
            model = model.to(device)
            model.eval()
        ######################
        seg_metrics = {"dice": {"SwinSAM-binary": DiceMetric(include_background=False, reduction='none'),
                                "SwinSAM-multi": DiceMetric(include_background=False, reduction='none'),
                                "Swin-GradCAM": DiceMetric(include_background=False, reduction='none'),
                                "Swin-HGI-SAM": DiceMetric(include_background=False, reduction='none'),
                                "UNet": DiceMetric(include_background=False, reduction='none')},
                       "iou":  {"SwinSAM-binary": MeanIoU(include_background=False, reduction='none'),
                                "SwinSAM-multi": MeanIoU(include_background=False, reduction='none'),
                                "Swin-GradCAM": MeanIoU(include_background=False, reduction='none'),
                                "Swin-HGI-SAM": MeanIoU(include_background=False, reduction='none'),
                                "UNet": MeanIoU(include_background=False, reduction='none')}
                       }

        best_thresholds = {"SwinSAM-binary": 0.06, "SwinSAM-multi": 0.1, "Swin-GradCAM": 0.8, "Swin-HGI-SAM": 0.06}
        # best_thresholds = {"SwinSAM-binary": find_best_threshold(np.arange(0.03, 0.15, 0.01), val_ds, models["SwinSAM-binary"], "SwinSAM-binary", device),
        #                    "SwinSAM-multi": find_best_threshold(np.arange(0.07, 0.15, 0.01), val_ds, models["SwinSAM-multi"], "SwinSAM-multi", device),
        #                    "Swin-GradCAM": find_best_threshold(np.arange(0.5, 1, 0.1), val_ds, models["Swin-GradCAM"], "Swin-GradCAM", device),
        #                    "Swin-HGI-SAM": find_best_threshold(np.arange(0.03, 0.15, 0.01), val_ds, models["Swin-HGI-SAM"], "Swin-HGI-SAM", device)}

        confusion_matrices = {"SwinSAM-binary": ConfusionMatrix(), "SwinSAM-multi": ConfusionMatrix(),
                              "Swin-HGI-SAM": ConfusionMatrix(), "UNet": ConfusionMatrix()}

        pbar_test = tqdm(val_ds, total=len(val_ds), leave=False)
        pbar_test.set_description(f'testing fold: {fold_number}')
        for x, mask, brain, y in pbar_test:
            x, mask, brain = x.to('cuda'), mask.to('cuda'), brain.to('cuda')

            p_mlcn_multi = model_mlcn_multi(deepcopy(x.unsqueeze(0)))
            mlcn_predictin_multi = torch.round(torch.sigmoid(p_mlcn_multi)).view(-1)

            p_mlcn = model_mlcn_binary(deepcopy(x.unsqueeze(0)))
            mlcn_predictin = torch.round(torch.sigmoid(p_mlcn))

            p_grad = model_grad(deepcopy(x.unsqueeze(0)))
            grad_prediction = torch.argmax(p_grad, dim=1)
            #############################################
            p_mask_unet = model_unet(x.unsqueeze(0)).detach()
            p_mask_unet = torch.softmax(p_mask_unet, dim=1)
            p_mask_unet_b = torch.argmax(deepcopy(p_mask_unet), dim=1)
            unet_prediction = torch.Tensor([1]) if p_mask_unet_b.sum() > 10 else torch.Tensor([0])
            unet_prediction = unet_prediction.to('cuda')

            cfm_mlcn_binary.add_prediction(mlcn_predictin, y[-1])
            cfm_mlcn_multi.add_prediction(mlcn_predictin_multi[-1:], y[-1])
            cfm_gradsam.add_prediction(grad_prediction, y[-1])
            cfm_unet.add_prediction(unet_prediction, y[-1])
            if y[-1] == 1:
                mask = to_onehot(m)
                ##########################################
                start_time = time.time()
                foregrounds = p_grad[:, 1].sum()
                foregrounds.backward()
                p_mask_grad = model_grad.attentional_segmentation_grad(brain)
                p_mask_grad_b = binarization_simple_thresholding(deepcopy(p_mask_grad), best_th_grad)
                p_mask_grad_b = to_onehot(p_mask_grad_b)
                gradsam_total_time += time.time() - start_time
                #####################################################
                start_time = time.time()
                p_mask_gradcam = model_grad.grad_cam_segmentation(x.unsqueeze(0), brain)
                p_mask_gradcam_b = binarization_simple_thresholding(deepcopy(p_mask_gradcam), best_th_gradcam)
                p_mask_gradcam_b = to_onehot(p_mask_gradcam_b)
                gradcam_total_time += time.time() - start_time
                ##################################################
                start_time = time.time()
                p_mask_mlcn_multi = model_mlcn_multi.attentional_segmentation(brain)
                p_mask_mlcn_multi_b = binarization_simple_thresholding(deepcopy(p_mask_mlcn_multi), best_th_mlcn_multi)
                p_mask_mlcn_multi_b = to_onehot(p_mask_mlcn_multi_b)
                mlcn_multi_total_time += time.time() - start_time
                ##################################################
                start_time = time.time()
                p_mask_mlcn = model_mlcn_binary.attentional_segmentation(brain)
                p_mask_mlcn_b = binarization_simple_thresholding(deepcopy(p_mask_mlcn), best_th_weight)
                p_mask_mlcn_b = to_onehot(p_mask_mlcn_b)
                mlcn_binary_total_time += time.time() - start_time
                ##############################################
                p_mask_unet_b = to_onehot(p_mask_unet_b)
                #############################################
                dice_grad(p_mask_grad_b.unsqueeze(0), mask.unsqueeze(0))
                iou_grad(p_mask_grad_b.unsqueeze(0), mask.unsqueeze(0))
                dice_weight(p_mask_mlcn_b.unsqueeze(0), mask.unsqueeze(0))
                iou_weight(p_mask_mlcn_b.unsqueeze(0), mask.unsqueeze(0))
                dice_mlcn_multi(p_mask_mlcn_multi_b.unsqueeze(0), mask.unsqueeze(0))
                iou_mlcn_multi(p_mask_mlcn_multi_b.unsqueeze(0), mask.unsqueeze(0))
                dice_unet(p_mask_unet_b.unsqueeze(0), mask.unsqueeze(0))
                iou_unet(p_mask_unet_b.unsqueeze(0), mask.unsqueeze(0))
                dice_gradcam(p_mask_gradcam_b.unsqueeze(0), mask.unsqueeze(0))
                iou_gradcam(p_mask_gradcam_b.unsqueeze(0), mask.unsqueeze(0))
                #############################################
                # brain_window_np = x[0].cpu().numpy()
                # mask_np = mask[1].cpu().numpy()
                # cv2.imshow('extra/samples/image.png', x.permute(1, 2, 0).cpu().numpy())
                # cv2.imshow('extra/samples/brain-w.png', x[0].cpu().numpy())
                # # cv2.imshow('extra/samples/subdural-w.png', x[1].cpu().numpy())
                # # cv2.imshow('extra/samples/bone-w.png', x[2].cpu().numpy())
                # cv2.imshow('extra/samples/brain.png', brain.squeeze().cpu().numpy())
                # cv2.imshow('extra/samples/mask.png', mask[1].cpu().numpy())
                #
                # cv2.imshow('extra/samples/pred-grad.png', p_mask_grad_b[1].cpu().numpy())
                # cv2.imshow('extra/samples/pred-mlcn-binary.png', p_mask_mlcn_b[1].cpu().numpy())
                # cv2.imshow('extra/samples/pred-mlcn-multi.png', p_mask_mlcn_multi_b[1].cpu().numpy())
                # cv2.imshow('extra/samples/pred-unet.png', p_mask_unet_b[1].cpu().numpy())
                # cv2.imshow('extra/samples/pred-gradcam.png', p_mask_gradcam_b[1].cpu().numpy())
                #
                # cv2.imwrite('extra/samples/brain-window.png', brain_window_np * 256)
                # method('extra/samples/pred-grad.png', brain_window_np, mask_np, p_mask_grad_b[1].cpu().numpy())
                # method('extra/samples/pred-mlcn-binary.png', brain_window_np, mask_np, p_mask_mlcn_b[1].cpu().numpy())
                # method('extra/samples/pred-mlcn-multi.png', brain_window_np, mask_np, p_mask_mlcn_multi_b[1].cpu().numpy())
                # method('extra/samples/pred-unet.png', brain_window_np, mask_np, p_mask_unet_b[1].cpu().numpy())
                # method('extra/samples/pred-gradcam.png', brain_window_np, mask_np, p_mask_gradcam_b[1].cpu().numpy())
                #
                # cv2.imshow('extra/samples/attention-map-grad.png', p_mask_grad[0].cpu().numpy())
                # cv2.imshow('extra/samples/attention-map-mlcn-binary.png', p_mask_mlcn[0].cpu().numpy())
                # cv2.imshow('extra/samples/attention-map-mlcn-multi.png', p_mask_mlcn_multi[0].cpu().numpy())
                # cv2.imshow('extra/samples/gradcam-map.png', p_mask_gradcam[0].cpu().numpy())
                #
                # cv2.waitKey()

        cfm_mlcn_binary.compute_confusion_matrix()
        cfm_mlcn_multi.compute_confusion_matrix()
        cfm_gradsam.compute_confusion_matrix()
        cfm_unet.compute_confusion_matrix()
        print('fold', fold_number)
        # print('\ngrad: dice=', torch.std_mean(dice_grad.get_buffer()), 'iou=', torch.std_mean(iou_grad.get_buffer()), '\n',
        #       'accuracy=', cfm_gradsam.get_accuracy(), 'precision=', cfm_gradsam.get_precision(), 'recall', cfm_gradsam.get_recall_sensitivity(),
        #       'F1=', cfm_gradsam.get_f1_score(), 'Specificity=', cfm_gradsam.get_specificity(), "AUC=", cfm_gradsam.get_auc_score())
        print('avg time=', gradsam_total_time / num_samples)
        dices_grad.extend(list(dice_grad.get_buffer().view(-1).cpu().numpy()))
        ious_grad.extend(list(iou_grad.get_buffer().view(-1).cpu().numpy()))

        # print('\nmlcn binary: dice=', torch.std_mean(dice_weight.get_buffer()), 'iou=', torch.std_mean(iou_weight.get_buffer()), '\n',
        #       'accuracy=', cfm_mlcn_binary.get_accuracy(), 'precision=', cfm_mlcn_binary.get_precision(), 'recall', cfm_mlcn_binary.get_recall_sensitivity(),
        #       'F1=', cfm_mlcn_binary.get_f1_score(), 'Specificity=', cfm_mlcn_binary.get_specificity(), "AUC=", cfm_mlcn_binary.get_auc_score())
        print('avg time=', mlcn_binary_total_time / num_samples)
        dices_SwinSAM.extend(list(dice_weight.get_buffer().view(-1).cpu().numpy()))
        ious_weight.extend(list(iou_weight.get_buffer().view(-1).cpu().numpy()))

        # print('\nmlcn multi: dice=', torch.std_mean(dice_mlcn_multi.get_buffer()), 'iou=', torch.std_mean(iou_mlcn_multi.get_buffer()), '\n',
        #       'accuracy=', cfm_mlcn_multi.get_accuracy(), 'precision=', cfm_mlcn_multi.get_precision(), 'recall', cfm_mlcn_multi.get_recall_sensitivity(),
        #       'F1=', cfm_mlcn_multi.get_f1_score(), 'Specificity=', cfm_mlcn_multi.get_specificity(), "AUC=", cfm_mlcn_multi.get_auc_score())
        print('avg time=', mlcn_multi_total_time / num_samples)
        dices_mlcn_multi.extend(list(dice_mlcn_multi.get_buffer().view(-1).cpu().numpy()))
        ious_mlcn_multi.extend(list(iou_mlcn_multi.get_buffer().view(-1).cpu().numpy()))

        # print('\nunet: dice=', torch.std_mean(dice_unet.get_buffer()), 'iou=', torch.std_mean(iou_unet.get_buffer()), '\n',
        #       'accuracy=', cfm_unet.get_accuracy(), 'precision=', cfm_unet.get_precision(), 'recall', cfm_unet.get_recall_sensitivity(),
        #       'F1=', cfm_unet.get_f1_score(), 'Specificity=', cfm_unet.get_specificity(), "AUC=", cfm_unet.get_auc_score())
        print('avg time=', unet_total_time / num_samples)
        dices_unet.extend(list(dice_unet.get_buffer().view(-1).cpu().numpy()))
        ious_unet.extend(list(iou_unet.get_buffer().view(-1).cpu().numpy()))

        # print('\ngrad cam: dice=', torch.nanmean(dice_gradcam.get_buffer()), np.nanstd(dice_gradcam.get_buffer().cpu().numpy()),
        #       'iou=', torch.nanmean(iou_gradcam.get_buffer()), np.nanstd(iou_gradcam.get_buffer().cpu().numpy()))
        print('avg time=', gradcam_total_time / num_samples)
        dices_gradcam.extend(list(dice_gradcam.get_buffer().view(-1).cpu().numpy()))
        ious_gradcam.extend(list(iou_gradcam.get_buffer().view(-1).cpu().numpy()))

    res = {'dice_mlcn_binary': dices_SwinSAM, 'dice_mlcn_multi': dices_mlcn_multi, 'dice_grad': dices_grad, 'dice_unet': dices_unet, 'dice_gradcam': dices_gradcam,
           'iou_mlcn_binary': ious_weight, 'iou_mlcn_multi': ious_mlcn_multi, 'iou_grad': ious_grad, 'iou_unet': ious_unet, 'iou_gradcam': ious_gradcam}
    df = pd.DataFrame(res)
    df.to_csv(r'extra\results\results.csv')

    print('overall subject based:')
    print('dice_mlcn_binary', torch.std_mean(torch.tensor(dices_SwinSAM)))
    print('dice_mlcn_multi', torch.std_mean(torch.tensor(dices_mlcn_multi)))
    print('dice_grad', torch.std_mean(torch.tensor(dices_grad)))
    print('dices_unet', torch.std_mean(torch.tensor(dices_unet)))
    print('dices_gradcam', np.nanstd(np.array(dices_gradcam)), np.nanmean(np.array(dices_gradcam)))
    print('iou_mlcn_binary', torch.std_mean(torch.tensor(ious_weight)))
    print('ious_mlcn_multi', torch.std_mean(torch.tensor(ious_mlcn_multi)))
    print('ious_grad', torch.std_mean(torch.tensor(ious_grad)))
    print('iou_unet', torch.std_mean(torch.tensor(ious_unet)))
    print('ious_gradcam', np.nanstd(np.array(ious_gradcam)), np.nanmean(np.array(ious_gradcam)))


def method(name, brain, mask, pred):
    color_brain = np.stack(3 * [brain], axis=-1)
    color_pred = np.zeros_like(color_brain)
    color_pred[..., 2] = pred
    color_mask = np.zeros_like(color_brain)
    color_mask[..., 1] = mask

    out = color_brain + color_mask + color_pred
    out = (out - out.min()) / (out.max() - out.min()) * 256
    cv2.imwrite(name, out)


def find_best_threshold(grid_range, val_ds, model, model_name, device='cuda'):
    best_dice = -1
    best_th = 0
    for th in grid_range:
        # we use dice metric to find the best threshold using grid-search
        dice_val = DiceMetric(include_background=False, reduction='mean')

        pbar_val = tqdm(val_ds, total=len(val_ds), leave=False)
        pbar_val.set_description(f'finding the best threshold for {model_name}')
        for x, mask, brain, y in pbar_val:
            if y[-1] == 1:  # we only consider samples that has a mask
                x, mask, brain = x.to(device), mask.to(device), brain.to(device)
                if model_name == "Swin-HGI-SAM":
                    p = model(x.unsqueeze(0))
                    foregrounds = p[:, 1].sum()
                    foregrounds.backward()
                    p_mask = model.attentional_segmentation_grad(brain)
                elif model_name in ["SwinSAM-binary", "SwinSAM-multi"]:
                    model(x.unsqueeze(0))
                    p_mask = model.attentional_segmentation(brain)
                else:  # Swin-GradCAM
                    p_mask = model.grad_cam_segmentation(x.unsqueeze(0), brain)
                p_mask = binarization_simple_thresholding(p_mask, th)
                p_mask = to_onehot(p_mask)
                mask = to_onehot(mask)
                dice_val(p_mask.unsqueeze(0), mask.unsqueeze(0))
        d = dice_val.aggregate()
        if d > best_dice:
            best_dice = d
            best_th = th
        else:
            break
    return best_th


if __name__ == '__main__':
    main()
    # m = SwinWeak(3, 6)
    #
    # swin = timm.models.swin_base_patch4_window12_384_in22k(in_chans=3, num_classes=6)
    # swin.load_state_dict(torch.load(r"C:\rsna-ich\Good weights\backup\Focal-100-checkpoint-2.pt"))
    #
    # for name, param in m.swin.state_dict().items():
    #     param.copy_(swin.state_dict()[name])
    # for name, param in m.head.state_dict().items():
    #     param.copy_(swin.head.state_dict()[name])
    # torch.save(m.state_dict(), r"D:\Projects\MELBA\extra\weights\multi.pth")
    # load_model(m.swin)
    # print(m)
