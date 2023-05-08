from helpers.utils import *
import timm
from collections import OrderedDict
from pytorch_grad_cam import GradCAM


class SwinWeak(nn.Module):
    def __init__(self, in_ch, num_classes):
        """
        A Swin transformer class that extracts attention weights and their respective gradients to do weakly supervised segmentation

        :param in_ch: number of input image channels
        :param num_classes: number of classes in classification task
        """
        super().__init__()
        self.swin = timm.models.swin_base_patch4_window12_384_in22k(in_chans=in_ch, num_classes=-1, pretrained=True)
        self.head = nn.Linear(1024, num_classes)

        # defining forward and backward hooks on attention weights to get the weights and their gradients
        self.attentions = OrderedDict()
        self.attentions_grads = OrderedDict()
        for ln, layer in enumerate(self.swin.layers):
            for bn, block in enumerate(layer.blocks):
                block.attn.softmax.register_forward_hook(get_attentions(self.attentions, f'{ln}_{bn}'))
                block.attn.softmax.register_full_backward_hook(get_attentions_grads(self.attentions_grads, f'{ln}_{bn}'))

    def forward(self, x):
        """
        classification forward

        :param x: input image
        :return: classification result
        """
        features = self.swin(x)
        return self.head(features)

    def attentional_segmentation(self, brainmask):
        """
        Swin-SAM: weak segmentation mask generation that only uses forward pass (MLCN paper)

        :param brainmask: brain-mask used to mask out predictions overlapping background and skull
        :return: segmentation map
        """
        mask = torch.ones_like(brainmask)
        for i in range(4):  # we use attention weights of all 4 layers
            mask *= _get_layer_attention_mask(self.attentions, 384, 4, 12, i, list(range(2)) if i != 2 else list(range(18)))
        mask = mask * brainmask

        mask = (mask - mask.min()) / (mask.max() - mask.min())  # 0-1 normalization of segmentation mask
        return mask

    def attentional_segmentation_grad(self, brainmask):
        """
        HGI-SAM: weak segmentation mask generation that uses both attention weights and their respective gradients

        :param brainmask: brain-mask used to mask out predictions overlapping background and skull
        :return: segmentation map
        """
        mask = torch.ones_like(brainmask)
        for i in range(3):  # we only use attention weights of first 3 layers
            mask *= _get_layer_attention_mask(self.attentions, 384, 4, 12, i, list(range(2)) if i != 2 else list(range(18)),
                                              attentions_grad=self.attentions_grads)
        mask = mask * brainmask

        mask = (mask - mask.min()) / (mask.max() - mask.min())  # 0-1 normalization of segmentation mask
        return mask

    def grad_cam_segmentation(self, x, brain):
        """
        Swin-GradCAM: weak segmentation mask generation that uses GradCAM technique

        :param x: input image
        :param brain: brain-mask used to mask out predictions overlapping background and skull
        :return: segmentation map
        """
        # defining the last block norm layer as the GradCAM target layer
        gcam_layer = GradCAM(model=self, target_layers=[self.swin.layers[-1].blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform(12, 12))
        gcam_map = torch.tensor(gcam_layer(x), device=brain.device, dtype=brain.dtype)
        gcam_map *= brain

        gcam_map = (gcam_map - gcam_map.min()) / (gcam_map.max() - gcam_map.min())  # 0-1 normalization of segmentation mask
        return gcam_map


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    A window reverse method that constructs original image by concatenating windows.

    :param windows: input tensor in the format of (num_windows*B, window_size, window_size, C)
    :param window_size: Window size
    :param H: Height of image
    :param W: Width of image
    :return: window reversed tensor in the format of (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_attentions(att_dict, layer_block):
    """
    helper function used to get each block's attention weights

    :param att_dict: the dictionary in which we save the weight
    :param layer_block: the key for saving the weight in dictionary
    """
    def hook(model, inp, out):
        att_dict[layer_block] = out.detach()

    return hook


def get_attentions_grads(att_dict, layer_block):
    """
    helper function used to get the gradient of each block's attention weight

    :param att_dict: the dictionary in which we save the gradient
    :param layer_block: the key for saving the gradient in dictionary
    """
    def hook(model, inp, out):
        att_dict[layer_block] = inp[0].detach()

    return hook


def _get_layer_attention_mask(attentions: dict, img_size: int, patch_size: int, window_size: int, layer: int, blocks: list,
                              attentions_grad=None):
    """
    a method that produces the attention map using attention weights and their respective gradients

    :param attentions: attention weights
    :param img_size: size of input image
    :param patch_size: size of patch (4 in our Swin-Transformer Base)
    :param window_size: size of window (12 in our Swin-Transformer Base)
    :param layer: layer number
    :param blocks: list of block numbers that are being used in attention generation
    :param attentions_grad: gradient of attention weights
    :return:
    """
    total_tokens = img_size // patch_size
    layer_tokens = total_tokens // (2 ** layer)
    shift_size = window_size // 2

    avg_pooling = nn.AdaptiveAvgPool2d((1, window_size * window_size))
    masks = []
    for block_index in blocks:
        att = attentions[f'{layer}_{block_index}']
        if attentions_grad is not None:
            # head-wise gradient infused weighting of attention weights
            a, b, c, d = att.shape
            weight = torch.norm(attentions_grad[f'{layer}_{block_index}'], dim=[2, 3])
            weight = weight.view(a, b, 1, 1)
            att = torch.mean(att * weight, dim=1)
        else:
            # simple mean over heads
            att = torch.mean(att, dim=1)

        # pool over target sequences
        mask = avg_pooling(att).reshape(-1, window_size * window_size)
        # window reverse
        mask = mask.reshape(-1, window_size, window_size, 1)
        mask = window_reverse(mask, window_size, layer_tokens, layer_tokens)
        # reverse shift for odd blocks
        if block_index % 2 == 1:
            mask = torch.roll(mask, shifts=(shift_size, shift_size), dims=(1, 2))

        mask = mask.permute(0, 3, 1, 2)
        masks.append(mask)

    # multiplying attention maps of every two consecutive blocks
    final_masks = []
    for i in range(0, len(masks), 2):
        swin_block_mask = masks[i] * masks[i + 1]
        final_masks.append(swin_block_mask)

    # getting average over attention maps (used only in layer 2 which has 18 blocks)
    all_masks = torch.stack(final_masks)
    final_mask = torch.mean(all_masks, dim=0)

    # 0-1 normalization
    final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())

    # bilinear interpolation
    final_mask = F.interpolate(final_mask, size=(img_size, img_size), mode='bilinear').squeeze()
    return final_mask
