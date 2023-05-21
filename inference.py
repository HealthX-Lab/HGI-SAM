import torch
from helpers.dataset import *
from models.swin_weak import SwinWeak
import cv2
from helpers.utils import *
import numpy as np
from monai.metrics import DiceMetric, MeanIoU
from models.unet import UNet
from torch.utils.data import Subset
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import cv2
import argparse
import json
from monai.networks.nets import SwinUNETR
from scipy.stats import sem


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
                      "UNet": config_dict["UNet_dir_path"],
                      "SwinUNETR": config_dict["SwinUNETR_dir_path"]}

    csv_seg_results_path = config_dict["csv_seg_results_path"]
    save_visualizations = str_to_bool(config_dict["save_visualizations"])
    save_visualizations_dir = config_dict["save_visualizations_dir"]
    if save_visualizations:
        assert os.path.isdir(save_visualizations_dir)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ds = PhysioNetICHDataset(data_path, windows=[(80, 340), (700, 3200)])
    # dictionary holding segmentation results
    seg_results = {"dice_SwinSAM-binary": [], "dice_SwinSAM-multi": [], "dice_Swin-GradCAM": [], "dice_Swin-HGI-SAM": [], "dice_UNet": [], "dice_SwinUNETR": [],
                   "iou_SwinSAM-binary": [], "iou_SwinSAM-multi": [], "iou_Swin-GradCAM": [], "iou_Swin-HGI-SAM": [], "iou_UNet": [], "iou_SwinUNETR": []}

    # iterating over folds
    for fold_number in [0, 1, 2, 3, 4]:
        with open(os.path.join(extra_path, "folds_division", f"fold{fold_number}.pt"), 'rb') as test_indices_file:
            test_indices = pickle.load(test_indices_file)
        test_ds = Subset(ds, test_indices)  # used for evaluating the model
        val_indices = [x for x in range(0, len(ds.labels)) if x not in test_indices]
        val_ds = Subset(ds, val_indices)  # used in weakly supervised models only to find the optimum threshold for binarization

        models = {"SwinSAM-binary": SwinWeak(in_ch=3, num_classes=1, pretrained=False),  # here, we use sigmoid + rounding to do binary classification
                  "SwinSAM-multi": SwinWeak(in_ch=3, num_classes=6, pretrained=False),
                  # in following, we use softmax + argmax to do binary classification,
                  "Swin-GradCAM": SwinWeak(in_ch=3, num_classes=2, pretrained=False),
                  "Swin-HGI-SAM": SwinWeak(in_ch=3, num_classes=2, pretrained=False),
                  "UNet": UNet(in_ch=3, num_classes=2, embed_dims=[24, 48, 96, 192]),
                  "SwinUNETR": SwinUNETR(img_size=(384, 384), in_channels=3, out_channels=2, spatial_dims=2)}

        for model_name, model in models.items():
            # only for UNet and SwinUNETR models, we have to load the weight corresponding to the fold
            # thus, the path in the config file for UNet model is the path to the directory that contains the weights
            if model_name in ["UNet", "SwinUNETR"]:
                load_model(model, os.path.join(models_weights[model_name], f"{model_name}-DiceCELoss-fold{fold_number}.pth"))
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
                                "UNet": DiceMetric(include_background=False, reduction='none'),
                                "SwinUNETR": DiceMetric(include_background=False, reduction='none')},
                       "iou":  {"SwinSAM-binary": MeanIoU(include_background=False, reduction='none'),
                                "SwinSAM-multi": MeanIoU(include_background=False, reduction='none'),
                                "Swin-GradCAM": MeanIoU(include_background=False, reduction='none'),
                                "Swin-HGI-SAM": MeanIoU(include_background=False, reduction='none'),
                                "UNet": MeanIoU(include_background=False, reduction='none'),
                                "SwinUNETR": MeanIoU(include_background=False, reduction='none')}
                       }

        # thresholds for prediction mask binarization. Uncomment bellow for actual grid-search (slow)
        # best_thresholds = {"SwinSAM-binary": 0.06, "SwinSAM-multi": 0.1, "Swin-GradCAM": 0.8, "Swin-HGI-SAM": 0.06}
        best_thresholds = {"SwinSAM-binary": find_best_threshold(np.arange(0.03, 0.15, 0.01), val_ds, models["SwinSAM-binary"], "SwinSAM-binary", device),
                           "SwinSAM-multi": find_best_threshold(np.arange(0.07, 0.15, 0.01), val_ds, models["SwinSAM-multi"], "SwinSAM-multi", device),
                           "Swin-GradCAM": find_best_threshold(np.arange(0.4, 1, 0.1), val_ds, models["Swin-GradCAM"], "Swin-GradCAM", device),
                           "Swin-HGI-SAM": find_best_threshold(np.arange(0.03, 0.15, 0.01), val_ds, models["Swin-HGI-SAM"], "Swin-HGI-SAM", device)}

        confusion_matrices = {"SwinSAM-binary": ConfusionMatrix(), "SwinSAM-multi": ConfusionMatrix(),
                              "Swin-HGI-SAM": ConfusionMatrix(), "UNet": ConfusionMatrix(), "SwinUNETR": ConfusionMatrix()}

        pbar_test = tqdm(enumerate(test_ds), total=len(test_ds), leave=False)
        pbar_test.set_description(f'testing fold: {fold_number}')
        for i, (x, mask, brain, y) in pbar_test:
            x, mask, brain = x.to('cuda'), mask.to('cuda'), brain.to('cuda')

            logits = {}  # to store the model predictions of input x
            for model_name, model in models.items():
                logits[model_name] = model(deepcopy(x.unsqueeze(0)))

            pred_swin_sam_binary = torch.round(torch.sigmoid(logits["SwinSAM-binary"]))
            pred_swin_sam_multi = torch.round(torch.sigmoid(logits["SwinSAM-multi"])).view(-1)
            pred_swin_hgi_sam = torch.argmax(torch.softmax(logits["Swin-HGI-SAM"], dim=1), dim=1)
            predmask_unet = torch.argmax(torch.softmax(logits["UNet"], dim=1), dim=1)
            pred_unet = torch.Tensor([1]) if predmask_unet.sum() > 10 else torch.Tensor([0])
            pred_unet = pred_unet.to(device)
            predmask_swinunetr = torch.argmax(torch.softmax(logits["SwinUNETR"], dim=1), dim=1)
            pred_swinunetr = torch.Tensor([1]) if predmask_swinunetr.sum() > 10 else torch.Tensor([0])
            pred_swinunetr = pred_swinunetr.to(device)

            confusion_matrices["SwinSAM-binary"].add_prediction(pred_swin_sam_binary, y[-1:])
            confusion_matrices["SwinSAM-multi"].add_prediction(pred_swin_sam_multi[-1:], y[-1:])
            confusion_matrices["Swin-HGI-SAM"].add_prediction(pred_swin_hgi_sam, y[-1:])
            confusion_matrices["UNet"].add_prediction(pred_unet, y[-1:])
            confusion_matrices["SwinUNETR"].add_prediction(pred_swinunetr, y[-1:])

            # measuring segmentation metrics only for slices that have a mask
            predmasks = {"UNet": predmask_unet, "SwinUNETR": predmask_swinunetr}
            predmasks_onehot = {}
            if y[-1] == 1:
                mask_onehot = to_onehot(mask)

                # computing all prediction masks
                hgi_foregrounds = logits["Swin-HGI-SAM"][:, 1].sum()
                hgi_foregrounds.backward()
                predmasks["SwinSAM-binary"] = models["SwinSAM-binary"].attentional_segmentation(brain)
                predmasks["SwinSAM-multi"] = models["SwinSAM-multi"].attentional_segmentation(brain)
                predmasks["Swin-GradCAM"] = models["Swin-GradCAM"].grad_cam_segmentation(x.unsqueeze(0), brain)
                predmasks["Swin-HGI-SAM"] = models["Swin-HGI-SAM"].attentional_segmentation_grad(brain)
                # binarization of predmasks
                for model_name, predmask in predmasks.items():
                    if model_name in ["UNet", "SwinUNETR"]:
                        predmask_binary = predmasks[model_name]
                    else:
                        predmask_binary = binarization_simple_thresholding(deepcopy(predmasks[model_name]), best_thresholds[model_name])
                    predmask_binary_onehot = to_onehot(predmask_binary)
                    predmasks_onehot[model_name] = predmask_binary_onehot

                # computing segmentation metrics
                for metric_models in seg_metrics.values():
                    for model_name, metric in metric_models.items():
                        metric(predmasks_onehot[model_name].unsqueeze(0), mask_onehot.unsqueeze(0))

                # store segmentation maps
                if save_visualizations:
                    sample_dir_path = os.path.join(save_visualizations_dir, f'fold{fold_number}-sample{i:03d}')
                    if not os.path.isdir(sample_dir_path):
                        os.mkdir(sample_dir_path)
                    input_image = x.permute(1, 2, 0).cpu().numpy()
                    brain_window = x[0].cpu().numpy()
                    subdural_window = x[1].cpu().numpy()
                    bone_window = x[2].cpu().numpy()
                    # storing input image channels
                    cv2.imwrite(os.path.join(sample_dir_path, "input_image.png"), input_image * 256)
                    cv2.imwrite(os.path.join(sample_dir_path, "brain_window.png"), brain_window * 256)
                    cv2.imwrite(os.path.join(sample_dir_path, "subdural_window.png"), subdural_window * 256)
                    cv2.imwrite(os.path.join(sample_dir_path, "bone_window.png"), bone_window * 256)
                    # storing predicted segmentations along with ground truth
                    for model_name, predmask_onehot in predmasks_onehot.items():
                        save_segmentation_visualization(os.path.join(sample_dir_path, f"{model_name}.png"),
                                                        brain_window, mask.cpu().numpy(), predmask_onehot[1].cpu().numpy())

        # computing confusion matrices for detection metrics
        for cfm in confusion_matrices.values():
            cfm.compute_confusion_matrix()

        # printing results
        print(f'Fold {fold_number}:')
        print('Detection results:')
        for model_name, cfm in confusion_matrices.items():
            print(f'{model_name}:', end='\t')
            print(f'\t Accuracy={cfm.get_accuracy():.3f} \t Precision={cfm.get_precision():.3f} \t Recall={cfm.get_recall_sensitivity():.3f} '
                  f'\t F1={cfm.get_f1_score():.3f} \t Specificity={cfm.get_specificity():.3f} \t AUC={cfm.get_auc_score():.3f}')

        print('\nSegmentation results:')
        for metric_name, metric_models in seg_metrics.items():
            print(f'{metric_name}:', end='\t')
            for model_name, metric in metric_models.items():
                buffer = metric.get_buffer()
                buffer_np = buffer.cpu().numpy()
                seg_results[f'{metric_name}_{model_name}'].extend(list(buffer_np))
                print(f'{model_name}={torch.nanmean(buffer):.3f} +/- {np.nanstd(buffer_np):.3f} ({sem(buffer_np, nan_policy="omit")})', end='\t')
            print()

    seg_results_df = pd.DataFrame(seg_results)
    if os.path.isfile(csv_seg_results_path):
        os.remove(csv_seg_results_path)
    seg_results_df.to_csv(csv_seg_results_path)

    print('\n', 20 * '#')
    print('Overall subject-based segmentation results:')
    for metric_name, metric_models in seg_metrics.items():
        print(f'{metric_name}:', end='\t')
        for model_name, metric in metric_models.items():
            buffer_np = np.array(seg_results[f'{metric_name}_{model_name}'])
            buffer = torch.tensor(buffer_np)
            print(f'{model_name}={torch.nanmean(buffer):.3f} +/- {np.nanstd(buffer_np):.3f} ({sem(buffer_np, nan_policy="omit")})', end='\t')
        print()


def save_segmentation_visualization(path, brain_window, mask, predmask):
    """
    A method that stores a visualization of segmentation maps on top of a brain-window image,
    where the ground-truth will be green and prediction will be red.

    :param path: path to save the image
    :param brain_window: brain-window of the input image
    :param mask: ground-truth map
    :param predmask: prediction map
    """
    color_brain = np.stack(3 * [brain_window], axis=-1)
    color_pred = np.zeros_like(color_brain)
    color_pred[..., 2] = predmask
    color_mask = np.zeros_like(color_brain)
    color_mask[..., 1] = mask

    out = color_brain + color_mask + color_pred
    out = (out - out.min()) / (out.max() - out.min()) * 256
    cv2.imwrite(path, out)


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
