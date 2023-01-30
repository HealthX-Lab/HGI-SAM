import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
from copy import deepcopy
from tqdm import tqdm
import statistics
import timm
from torchvision.transforms import GaussianBlur
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.unet import ConvBlock


class SwinWeak(nn.Module):
    def __init__(self, num_classes, in_ch):
        super().__init__()
        self.swin_encoder = timm.models.swin_base_patch4_window12_384(in_chans=in_ch)

        self.attentions = {}
        for ln, layer in enumerate(self.swin_encoder.layers):
            for bn, block in enumerate(layer.blocks):
                block.attn.softmax.register_forward_hook(get_attentions(self.attentions, f'{ln}_{bn}'))

        self.features_layer_norm = nn.LayerNorm(1024)
        self.head = Head(1024, num_classes)

    def forward(self, x):
        x = self.swin_encoder.forward_features(x)
        x = self.features_layer_norm(x)

        b, hw, ch = x.shape
        h = w = int(np.sqrt(hw))
        x = x.reshape(b, h, w, ch)  # b, h, w, ch
        x = x.permute(0, 3, 1, 2)  # b, ch, h, w

        return self.head(x)


class Head(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.head = nn.Sequential(ConvBlock(in_ch, 128, 3),
                                  nn.Flatten(1),
                                  nn.Linear(12 * 12 * 128, 128),
                                  nn.BatchNorm1d(128),
                                  nn.Dropout(0.2),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(x)


def reshape_transform(height, width):
    def reshape(tensor):
        result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return reshape


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_attentions(att_dict, layer_block):
    def hook(model, inp, out):
        att_dict[layer_block] = out.detach()

    return hook


def get_attention_mask(attentions, g_cam=None, img_size=384, patch_size=4, window_size=12, device='cpu'):
    func = _get_layer_attention_mask
    softmax_layer = nn.Softmax(dim=1)

    mask0 = func(attentions, img_size, patch_size, window_size, 0, list(range(2)), device)
    mask1 = func(attentions, img_size, patch_size, window_size, 1, list(range(2)), device)
    mask2 = func(attentions, img_size, patch_size, window_size, 2, list(range(18)), device)
    mask3 = func(attentions, img_size, patch_size, window_size, 3, list(range(2)), device)
    # cv2.imshow('last', mask3.cpu().numpy())
    # cv2.imshow('second last', mask2.cpu().numpy())

    if g_cam is None:
        f_mask = mask3 * mask2 * mask1 * mask0
    else:
        g_cam = torch.tensor(g_cam, device=device)
        f_mask = g_cam * mask2 * mask1 * mask0

    H, W = f_mask.shape
    softmax = f_mask
    softmax = softmax.reshape(1, -1)
    softmax = softmax_layer(softmax)
    softmax = softmax.reshape(H, W)
    softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())
    # cv2.imwrite(r'C:\rsna-ich\sample images\softmax.bmp', 255 * softmax.cpu().numpy())

    return softmax


def _get_layer_attention_mask(attentions: dict, img_size: int, patch_size: int, window_size: int,
                              layer: int, blocks: list, device='cpu'):
    total_tokens = img_size // patch_size
    layer_tokens = total_tokens // (2 ** layer)
    shift_size = window_size // 2

    avg_pooling = nn.AdaptiveAvgPool2d((1, window_size * window_size))
    masks = []
    for block_index in blocks:
        att = attentions[f'{layer}_{block_index}']
        att = torch.mean(att, dim=1)  # mean over heads
        mask = avg_pooling(att).reshape(-1, window_size * window_size)  # pool over target sequences

        mask = mask.reshape(-1, window_size, window_size, 1)
        mask = window_reverse(mask, window_size, layer_tokens, layer_tokens)
        # if layer == 0:
        #     print(mask.shape)
        #     x = mask.squeeze().cpu().numpy()
        #     x = (x - x.min()) / (x.max() - x.min())
        #     cv2.imwrite(rf'C:\rsna-ich\sample images\maps\w{block_index}.bmp', x * 255)
        if block_index % 2 == 1:  # reverse shift for odd blocks
            mask = torch.roll(mask, shifts=(shift_size, shift_size), dims=(1, 2))
            # if layer == 0:
            #     x = mask.squeeze().cpu().numpy()
            #     x = (x - x.min()) / (x.max() - x.min())
            #     cv2.imwrite(rf'C:\rsna-ich\sample images\maps\sw{block_index}.bmp', x * 255)
        mask = mask.permute(0, 3, 1, 2)
        masks.append(mask)

    final_masks = []
    for i in range(0, len(masks), 2):
        swin_block_mask = masks[i] * masks[i + 1]
        # if layer == 0:
        #     x = swin_block_mask.squeeze().cpu().numpy()
        #     x = (x - x.min()) / (x.max() - x.min())
        #     cv2.imwrite(r'C:\rsna-ich\sample images\maps\mult.bmp', x * 255)
        final_masks.append(swin_block_mask)

    all_masks = torch.stack(final_masks)
    final_mask = torch.mean(all_masks, dim=0)

    final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())

    # cv2.imwrite(rf'C:\rsna-ich\sample images\layer {layer} attention.bmp', 255 * final_mask.squeeze().cpu().numpy())
    final_mask = F.interpolate(final_mask, size=(img_size, img_size), mode='bilinear').squeeze()
    # cv2.imwrite(rf'C:\rsna-ich\sample images\layer {layer} interpolated.bmp', 255 * final_mask.cpu().numpy())
    return final_mask


def segmentation(model: torch.nn.Module, dataloader, threshold, show_images=False, device='cpu'):
    model.to(device)
    model.eval()

    # gcam0 = GradCAM(model=model, target_layers=[model.layers[0].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(96, 96))
    # gcam1 = GradCAM(model=model, target_layers=[model.layers[1].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(48, 48))
    # gcam2 = GradCAM(model=model, target_layers=[model.layers[2].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(24, 24))
    # gcam3 = GradCAM(model=model, target_layers=[model.layers[3].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(12, 12))

    confusion_matrix = ConfusionMatrix()
    attentions = {}
    if isinstance(model, nn.Sequential):
        backbone = model[0]
    else:
        backbone = model
    for ln, layer in enumerate(backbone.layers):
        for bn, block in enumerate(layer.blocks):
            block.attn.softmax.register_forward_hook(get_attentions(attentions, f'{ln}_{bn}'))

    dices = []
    ious = []
    for sample, mask, label in dataloader:
        attentions.clear()
        sample, mask = sample.to(device), mask.to(device)
        with torch.no_grad():
            output = model(sample)
            confusion_matrix(torch.sigmoid(output), label)

        dice, iou = np.nan, np.nan
        if mask.any():
            img = sample.squeeze()
            brain_window = img[0]
            mask = mask[0]

            # cv2.imwrite(r'C:\rsna-ich\sample images\sample.bmp', 255 * img.permute(1, 2, 0).cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\brain window.bmp', 255 * brain_window.cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\subdural window.bmp', 255 * img[1].cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\bone window.bmp', 255 * img[2].cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\mask.bmp', 255 * mask.cpu().numpy())
            # grayscale_cam = gcam3(input_tensor=sample)[0]
            # pred_mask = torch.tensor(grayscale_cam, device=device)

            grayscale_cam = None
            pred_mask = get_attention_mask(attentions, grayscale_cam, device=device)

            denoized_brain_window = torch.tensor(cv2.GaussianBlur(brain_window.cpu().numpy(), (9, 9),
                                                                  cv2.BORDER_DEFAULT), device=device)
            # cv2.imwrite(r'C:\rsna-ich\sample images\denoised brain window.bmp',
            #             255 * denoized_brain_window.cpu().numpy())
            pred_mask = pred_mask * denoized_brain_window

            # cv2.imshow('pred mask', pred_mask.cpu().numpy())
            # cv2.imwrite(r'C:\rsna-ich\sample images\pred mask.bmp', 255 * pred_mask.cpu().numpy())
            # cv2.imshow('attention map', pred_mask.cpu().numpy())

            pred_mask = binarization_simple_thresholding(pred_mask, threshold)

            # cv2.imwrite(r'C:\rsna-ich\sample images\threshold mask.bmp', 255 * pred_mask.cpu().numpy())

            dice = dice_metric(pred_mask, mask).item()
            confusion_matrix.dices.append(dice)
            # print('\n', dice)
            iou = intersection_over_union(pred_mask, mask).item()

            if show_images:
                TP = torch.where(mask > 0, 1, 0) * torch.where(pred_mask > 0, 1, 0)
                FP = torch.where(pred_mask > 0, 1, 0) * torch.where(mask > 0, 0, 1)
                FN = torch.where(mask > 0, 1, 0) * torch.where(pred_mask > 0, 0, 1)

                m = torch.stack([FN, TP, FP])
                m = m.permute(1, 2, 0).cpu().numpy().astype(np.float)

                gt_mask = deepcopy(brain_window)
                gt_mask[FN > 0] = 1

                predicted_mask = deepcopy(brain_window)
                predicted_mask[FP > 0] = 1

                correct_pred = deepcopy(brain_window)
                correct_pred[TP > 0] = 1

                segmentation_image = torch.stack([gt_mask, correct_pred, predicted_mask])
                segmentation_image = segmentation_image.permute(1, 2, 0).cpu().numpy()

                cv2.imshow('segmentation', segmentation_image)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\segmentation-combined.bmp', 255 * segmentation_image)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\brain_window.bmp', 255 * brain_window.cpu().numpy())
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\pmask-combined.bmp', 255 * m)
                # cv2.imshow('mask', m)
                #
                # grayscale_cam0 = gcam3(input_tensor=sample)[0]
                #
                # # grayscale_cam1 = gcam1(input_tensor=sample)[0]
                # # grayscale_cam2 = gcam2(input_tensor=sample)[0]
                # # grayscale_cam3 = gcam3(input_tensor=sample)[0]
                #
                # cv2.imshow('cam0', grayscale_cam0)
                # # cv2.imshow('cam1', grayscale_cam1)
                # # cv2.imshow('cam2', grayscale_cam2)
                # # cv2.imshow('cam3', grayscale_cam3)
                # # all_g = grayscale_cam3 * grayscale_cam2 * grayscale_cam1 * grayscale_cam0
                # # all_g = (all_g - all_g.min()) / (all_g.max() - all_g.min())
                # # cv2.imshow('all', all_g)
                # # cv2.imshow('mean', (grayscale_cam3 + grayscale_cam2 + grayscale_cam1 + grayscale_cam0) / 4)
                # grayscale_cam0 = (grayscale_cam0 * 255).astype(np.uint8)
                # ret, grayscale_cam0 = cv2.threshold(grayscale_cam0, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # grayscale_cam0 = torch.tensor(grayscale_cam0, device=device)
                # TP2 = torch.where(mask > 0, 1, 0) * torch.where(grayscale_cam0 > 0, 1, 0)
                # FP2 = torch.where(grayscale_cam0 > 0, 1, 0) * torch.where(mask > 0, 0, 1)
                # FN2 = torch.where(mask > 0, 1, 0) * torch.where(grayscale_cam0 > 0, 0, 1)
                #
                # m2 = torch.stack([FN2, TP2, FP2])
                # m2 = m2.permute(1, 2, 0).cpu().numpy().astype(np.float)
                #
                # gt_mask2 = deepcopy(brain_window)
                # gt_mask2[FN2 > 0] = 1
                #
                # predicted_mask2 = deepcopy(brain_window)
                # predicted_mask2[FP2 > 0] = 1
                #
                # correct_pred2 = deepcopy(brain_window)
                # correct_pred2[TP2 > 0] = 1
                #
                # segmentation_image2 = torch.stack([gt_mask2, correct_pred2, predicted_mask2])
                # segmentation_image2 = segmentation_image2.permute(1, 2, 0).cpu().numpy()
                # cv2.imshow('segmentation2', segmentation_image2)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\segmentation2-combined.bmp',
                #             255 * segmentation_image2)
                # cv2.imwrite(r'C:\rsna-ich\sample images\segmentations\pmask2-combined.bmp', 255 * m2)
                # cv2.imshow('mask2', m2)

                cv2.waitKey()

        dices.append(dice)
        ious.append(iou)
        # plt.hist(dices)
        # plt.show()
    mean_dice = np.nanmean(dices)
    mean_iou = np.nanmean(ious)

    confusion_matrix.compute_confusion_matrix()
    return mean_dice, mean_iou, confusion_matrix
