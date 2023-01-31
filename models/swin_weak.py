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
from torchvision.transforms import GaussianBlur
from collections import OrderedDict


class SwinWeak(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.swin_encoder = timm.models.swin_base_patch4_window12_384(in_chans=in_ch, num_classes=-1)

        self.attentions = OrderedDict()
        for ln, layer in enumerate(self.swin_encoder.layers):
            for bn, block in enumerate(layer.blocks):
                block.attn.softmax.register_forward_hook(get_attentions(self.attentions, f'{ln}_{bn}'))

        self.head = Head(1024, num_classes)

        self.gaussian_blur = GaussianBlur(9, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.swin_encoder.forward_features(x)

        b, hw, ch = x.shape
        h = w = int(np.sqrt(hw))
        x = x.reshape(b, h, w, ch)  # b, h, w, ch
        x = x.permute(0, 3, 1, 2)  # b, ch, h, w

        return self.head(x)

    def segmentation(self, x):
        y = self.forward(x)

        x = self.gaussian_blur(x.squeeze(1))
        mask = torch.ones_like(x)
        for i in range(4):
            mask *= _get_layer_attention_mask(self.attentions, 384, 4, 12, i, list(range(2)) if i != 2 else list(range(18)))

        mask = mask * x
        b, h, w = mask.shape
        softmax = x.reshape(1, -1)
        softmax = self.softmax(softmax)
        softmax = softmax.reshape(b, h, w)
        softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())
        return y, softmax


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


def _get_layer_attention_mask(attentions: dict, img_size: int, patch_size: int, window_size: int,
                              layer: int, blocks: list, device='cuda'):
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
