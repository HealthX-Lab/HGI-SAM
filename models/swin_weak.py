import torch

from utils.utils import *
import timm
from torchvision.transforms import GaussianBlur
from collections import OrderedDict
from models.unet import UNet
import cv2


class SwinWeak(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.swin = timm.models.swin_base_patch4_window12_384_in22k(in_chans=in_ch, num_classes=num_classes, pretrained=True)

        self.attentions = OrderedDict()
        self.attentions_grads = OrderedDict()
        for ln, layer in enumerate(self.swin.layers):
            for bn, block in enumerate(layer.blocks):
                block.attn.softmax.register_forward_hook(get_attentions(self.attentions, f'{ln}_{bn}'))
                block.attn.softmax.register_backward_hook(get_attentions_grads(self.attentions_grads, f'{ln}_{bn}'))

        self.gaussian_blur = GaussianBlur(9, 2)
        self.softmax = nn.Softmax(dim=1)

        self.refinement_unet = UNet(in_ch, 2, embed_dims=[24, 48, 96, 192])

    def forward(self, x):
        return self.swin(x)

    def attentional_segmentation(self, x):
        y = self.forward(x)

        x = self.blur_brain_window(x)
        mask = torch.ones_like(x)
        for i in range(4):
            mask *= _get_layer_attention_mask(self.attentions, 384, 4, 12, i, list(range(2)) if i != 2 else list(range(18)))
        mask = mask * x

        b, h, w = mask.shape
        softmax = mask.reshape(1, -1)
        softmax = self.softmax(softmax)
        softmax = softmax.reshape(b, h, w)
        softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())
        return y, softmax

    def attentional_segmentation_grad(self, x):
        x = self.blur_brain_window(x)
        mask = torch.ones_like(x)
        for i in range(4):
            mask *= _get_layer_attention_mask(self.attentions_grads, 384, 4, 12, i, list(range(2)) if i != 2 else [16, 17], type='grad')
        # mask = mask * x

        b, h, w = mask.shape
        softmax = mask.reshape(1, -1)
        softmax = self.softmax(softmax)
        softmax = softmax.reshape(b, h, w)
        softmax = (softmax - softmax.min()) / (softmax.max() - softmax.min())
        return softmax

    def blur_brain_window(self, x):
        return self.gaussian_blur(x[:, 0, :, :])

    def refinement_segmentation(self, x):
        return self.refinement_unet(x)


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


def get_attentions_grads(att_dict, layer_block):
    def hook(model, inp, out):
        att_dict[layer_block] = out[0].detach()

    return hook


def _get_layer_attention_mask(attentions: dict, img_size: int, patch_size: int, window_size: int,
                              layer: int, blocks: list, device='cuda', type='weight'):
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

        if block_index % 2 == 1:  # reverse shift for odd blocks
            mask = torch.roll(mask, shifts=(shift_size, shift_size), dims=(1, 2))

        mask = mask.permute(0, 3, 1, 2)
        masks.append(mask)

    final_masks = []
    for i in range(0, len(masks), 2):
        swin_block_mask = masks[i] * masks[i + 1]
        final_masks.append(swin_block_mask)

    all_masks = torch.stack(final_masks)
    final_mask = torch.mean(all_masks, dim=0)

    final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())

    final_mask = F.interpolate(final_mask, size=(img_size, img_size), mode='bilinear').squeeze()
    # cv2.imshow(f'l{layer}-{type}', final_mask.squeeze().cpu().numpy())
    return final_mask
