import timm
import torch.nn as nn
from models.unet import ConvBlock
from collections import OrderedDict
import numpy as np
import torch


class SwinUNetR(nn.Module):
    def __init__(self, num_classes, in_ch, embed_dims=None):
        super().__init__()
        if embed_dims is None:
            self.embed_dims = [128, 256, 512, 1024, 2048]
        else:
            self.embed_dims = embed_dims
        self.swin_encoder = timm.models.swin_base_patch4_window12_384(in_chans=in_ch)

        self.input_embeder = ConvBlock(in_ch=in_ch, out_ch=self.embed_dims[0], kernel_size=3)
        self.patch_upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_dims[0], out_channels=self.embed_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                                            nn.ConvTranspose2d(in_channels=self.embed_dims[0], out_channels=self.embed_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1))
        self.final_conv_block = ConvBlock(in_ch=self.embed_dims[1], out_ch=self.embed_dims[0], kernel_size=3)

        self.bottleneck_patch_merging = PatchMerging(input_resolution=(12, 12), dim=self.embed_dims[-2])

        self.swin_residuals = OrderedDict()
        for ln, layer in enumerate(self.swin_encoder.layers):
            layer.register_forward_hook(get_intermediate_residuals(self.swin_residuals, ln))

        self.residual_blocks1 = OrderedDict()
        for i in range(len(self.embed_dims)):
            self.residual_blocks1[f'res-block{i}'] = ConvBlock(in_ch=self.embed_dims[i], out_ch=self.embed_dims[i], kernel_size=3)
        self.residual_blocks1 = nn.ModuleDict(self.residual_blocks1)

        self.embed_dims.reverse()
        decoder = OrderedDict()
        for i in range(len(self.embed_dims) - 1):
            decoder[f'up-{i + 1}'] = nn.ConvTranspose2d(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            decoder[f'expansive-{i + 1}'] = ConvBlock(in_ch=self.embed_dims[i], out_ch=self.embed_dims[i + 1], kernel_size=3)
        self.decoder = nn.ModuleDict(decoder)

        self.head = nn.Conv2d(in_channels=self.embed_dims[-1], out_channels=num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        swin_features = self.swin_encoder.forward_features(x)  # b, hw, ch
        bottleneck = self.bottleneck_patch_merging(swin_features)  # b, hw/4, 2ch
        b, hw, ch = bottleneck.shape
        h = w = int(np.sqrt(hw))
        bottleneck = bottleneck.reshape(b, h, w, ch)  # b, h/2, w/2, 2ch
        bottleneck = bottleneck.permute(0, 3, 1, 2)  # b, 2ch, h/2, w/2

        residuals = []
        for i in range(len(self.embed_dims) - 1):
            residuals.append(self.residual_blocks1[f'res-block{i}'](self.swin_residuals[i]))

        f = self.residual_blocks1[f'res-block{len(self.embed_dims) - 1}'](bottleneck)

        residuals.reverse()
        for i in range(len(self.embed_dims) - 1):
            f = self.decoder[f'up-{i + 1}'](f)
            f = self.decoder[f'expansive-{i + 1}'](torch.cat((f, residuals[i]), dim=1))

        x = self.input_embeder(x)
        x = self.final_conv_block(torch.cat((x, self.patch_upsample(f)), dim=1))

        return self.head(x)


def get_intermediate_residuals(saved_dict, layer_num):
    def hook(model, inp, out):
        input_to_layer = inp[0]
        b, hw, ch = input_to_layer.shape
        h = w = int(np.sqrt(hw))
        saved_dict[layer_num] = input_to_layer.reshape(b, h, w, ch).permute(0, 3, 1, 2)  # b, ch, h, w

    return hook


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops