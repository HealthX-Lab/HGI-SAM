import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from utils.dataset import *
from utils.preprocessing import get_transform
from models.swin_weak import SwinWeak
import cv2
from utils.utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
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


def main():
    physio_path = "C:\\physio-ich"
    ds = PhysioNetICHDataset(physio_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384), return_brain=True)
    dices_weight = []
    ious_weight = []
    dices_mlcn_multi = []
    ious_mlcn_multi = []
    dices_grad = []
    ious_grad = []
    dices_unet = []
    ious_unet = []
    dices_gradcam = []
    ious_gradcam = []
    for f in [0, 1, 2, 3, 4]:
        fold_number = f
        with open(rf"extra/folds_division/fold{fold_number}.pt", 'rb') as test_indices_file:
            test_indices = pickle.load(test_indices_file)
        test_ds = Subset(ds, test_indices)
        val_indices = [x for x in range(0, len(ds.labels)) if x not in test_indices]
        val_ds = Subset(ds, val_indices)

        model_grad = SwinWeak(3, 2)
        load_model(model_grad, r"extra/weights/SwinWeak_CrossEntropyLoss-grad.pth")
        model_mlcn_binary = SwinWeak(3, 1, mlcn=True)
        load_model(model_mlcn_binary.swin, r"C:\rsna-ich\Good weights\backup\Focal-100-checkpoint-0.pt")
        model_mlcn_multi = SwinWeak(3, 6, mlcn=True)
        load_model(model_mlcn_multi.swin, r"C:\rsna-ich\Good weights\backup\Focal-100-checkpoint-2.pt")
        model_unet = UNet(3, 2, [24, 48, 96, 192])
        load_model(model_unet, rf'D:\Projects\MELBA\extra\weights\UNet-DiceCELoss-fold{fold_number}.pt')
        ###########################
        model_grad.cuda()
        model_grad.eval()
        model_mlcn_binary.cuda()
        model_mlcn_binary.eval()
        model_mlcn_multi.cuda()
        model_mlcn_multi.eval()
        model_unet.cuda()
        model_unet.eval()
        ######################
        dice_grad = DiceMetric(include_background=False, reduction='none')
        iou_grad = MeanIoU(include_background=False, reduction='none')
        dice_weight = DiceMetric(include_background=False, reduction='none')
        iou_weight = MeanIoU(include_background=False, reduction='none')
        dice_mlcn_multi = DiceMetric(include_background=False, reduction='none')
        iou_mlcn_multi = MeanIoU(include_background=False, reduction='none')
        dice_unet = DiceMetric(include_background=False, reduction='none')
        iou_unet = MeanIoU(include_background=False, reduction='none')
        dice_gradcam = DiceMetric(include_background=False, reduction='none')
        iou_gradcam = MeanIoU(include_background=False, reduction='none')

        best_th_grad = 0.06
        best_th_weight = 0.06
        best_th_mlcn_multi = 0.1
        best_th_gradcam = 0.8
        # best_dice_grad = 0
        # best_dice_weight = 0
        # best_dice_mlcn_multi = 0
        # best_dice_gradcam = 0
        # for th in np.arange(0.03, 0.15, 0.01):
        #     dice_val = DiceMetric(include_background=False, reduction='mean')
        #     for x, m, y, brain in tqdm(val_ds):
        #         if y[-1] == 1:
        #             x, m, brain = x.to('cuda'), m.to('cuda'), brain.to('cuda')
        #             p = model_grad(x.unsqueeze(0))
        #             foregrounds = p[:, 1].sum()
        #             foregrounds.backward()
        #             p_mask = model_grad.attentional_segmentation_grad(brain)
        #             p_mask = binarization_simple_thresholding(p_mask, th)
        #             p_mask = to_onehot(p_mask)
        #             mask = to_onehot(m)
        #             dice_val(p_mask.unsqueeze(0), mask.unsqueeze(0))
        #     d = dice_val.aggregate()
        #     print('grad th:', th, ': ', d)
        #     if d > best_dice_grad:
        #         best_dice_grad = d
        #         best_th_grad = th
        #     else:
        #         break
        # print('grad best:', best_dice_grad, best_th_grad)
        #
        # for th in np.arange(0.03, 0.15, 0.01):
        #     dice_val = DiceMetric(include_background=False, reduction='mean')
        #     for x, m, y, brain in tqdm(val_ds):
        #         if y[-1] == 1:
        #             x, m, brain = x.to('cuda'), m.to('cuda'), brain.to('cuda')
        #             model_mlcn_binary(x.unsqueeze(0))
        #             p_mask = model_mlcn_binary.attentional_segmentation(brain)
        #             p_mask = binarization_simple_thresholding(p_mask, th)
        #             p_mask = to_onehot(p_mask)
        #             mask = to_onehot(m)
        #             dice_val(p_mask.unsqueeze(0), mask.unsqueeze(0))
        #     d = dice_val.aggregate()
        #     print('mlcn binary th:', th, ': ', d)
        #     if d > best_dice_weight:
        #         best_dice_weight = d
        #         best_th_weight = th
        #     else:
        #         break
        #
        # print('mlcn binary best:', best_dice_weight, best_th_weight)
        #
        # for th in np.arange(0.07, 0.15, 0.01):
        #     dice_val = DiceMetric(include_background=False, reduction='mean')
        #     for x, m, y, brain in tqdm(val_ds):
        #         if y[-1] == 1:
        #             x, m, brain = x.to('cuda'), m.to('cuda'), brain.to('cuda')
        #             model_mlcn_multi(x.unsqueeze(0))
        #             p_mask = model_mlcn_multi.attentional_segmentation(brain)
        #             p_mask = binarization_simple_thresholding(p_mask, th)
        #             p_mask = to_onehot(p_mask)
        #             mask = to_onehot(m)
        #             dice_val(p_mask.unsqueeze(0), mask.unsqueeze(0))
        #     d = dice_val.aggregate()
        #     print('mlcn multi th:', th, ': ', d)
        #     if d > best_dice_mlcn_multi:
        #         best_dice_mlcn_multi = d
        #         best_th_mlcn_multi = th
        #     else:
        #         break
        #
        # print('mlcn multi best:', best_dice_mlcn_multi, best_th_mlcn_multi)
        #
        # for th in np.arange(0.5, 1, 0.1):
        #     dice_val = DiceMetric(include_background=False, reduction='mean')
        #     for x, m, y, brain in tqdm(val_ds):
        #         if y[-1] == 1:
        #             x, m, brain = x.to('cuda'), m.to('cuda'), brain.to('cuda')
        #             p_mask = model_grad.grad_cam_segmentation(x.unsqueeze(0), brain)
        #             p_mask = binarization_simple_thresholding(p_mask, th)
        #             p_mask = to_onehot(p_mask)
        #             mask = to_onehot(m)
        #             dice_val(p_mask.unsqueeze(0), mask.unsqueeze(0))
        #     d = dice_val.aggregate()
        #     print('gradcam th:', th, ': ', d)
        #     if d > best_dice_gradcam:
        #         best_dice_gradcam = d
        #         best_th_gradcam = th
        #     else:
        #         break
        #
        # print('gradcam best:', best_dice_gradcam, best_th_gradcam)

        num_samples = 0
        cfm_mlcn_binary = ConfusionMatrix()
        mlcn_binary_total_time = 0
        cfm_mlcn_multi = ConfusionMatrix()
        mlcn_multi_total_time = 0
        cfm_gradsam = ConfusionMatrix()
        gradsam_total_time = 0
        gradcam_total_time = 0
        cfm_unet = ConfusionMatrix()
        unet_total_time = 0
        for x, m, y, brain in tqdm(test_ds):
            if y[-1] == 0:
                continue
            num_samples += 1
            x, m, brain = x.to('cuda'), m.to('cuda'), brain.to('cuda')

            start_time = time.time()
            p_mlcn_multi = model_mlcn_multi(deepcopy(x.unsqueeze(0)))
            mlcn_predictin_multi = torch.round(torch.sigmoid(p_mlcn_multi)).view(-1)
            mlcn_predictin_multi += time.time() - start_time

            start_time = time.time()
            p_mlcn = model_mlcn_binary(deepcopy(x.unsqueeze(0)))
            mlcn_predictin = torch.round(torch.sigmoid(p_mlcn))
            mlcn_binary_total_time += time.time() - start_time

            start_time = time.time()
            p_grad = model_grad(deepcopy(x.unsqueeze(0)))
            grad_prediction = torch.argmax(p_grad, dim=1)
            end_t = time.time() - start_time
            gradsam_total_time += end_t
            gradcam_total_time += end_t
            #############################################
            start_time = time.time()
            p_mask_unet = model_unet(x.unsqueeze(0)).detach()
            p_mask_unet = torch.softmax(p_mask_unet, dim=1)
            p_mask_unet_b = torch.argmax(deepcopy(p_mask_unet), dim=1)
            unet_prediction = torch.Tensor([1]) if p_mask_unet_b.sum() > 10 else torch.Tensor([0])
            unet_prediction = unet_prediction.to('cuda')
            unet_total_time += time.time() - start_time

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
        dices_weight.extend(list(dice_weight.get_buffer().view(-1).cpu().numpy()))
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

    res = {'dice_mlcn_binary': dices_weight, 'dice_mlcn_multi': dices_mlcn_multi, 'dice_grad': dices_grad, 'dice_unet': dices_unet, 'dice_gradcam': dices_gradcam,
           'iou_mlcn_binary': ious_weight, 'iou_mlcn_multi': ious_mlcn_multi, 'iou_grad': ious_grad, 'iou_unet': ious_unet, 'iou_gradcam': ious_gradcam}
    df = pd.DataFrame(res)
    df.to_csv(r'extra\results\results.csv')

    print('overall subject based:')
    print('dice_mlcn_binary', torch.std_mean(torch.tensor(dices_weight)))
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


if __name__ == '__main__':
    main()
