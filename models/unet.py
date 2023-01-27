import torch.nn as nn
import torch
from collections import OrderedDict


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, embed_dims=None):
        super().__init__()
        if embed_dims is None:
            self.embed_dims = [32, 64, 128, 256, 512, 1024]
        else:
            self.embed_dims = embed_dims
        self.num_classes = num_classes

        encoder = OrderedDict()
        for i in range(len(self.embed_dims) - 1):
            encoder[f'contracting-{i + 1}'] = ConvBlock(in_ch=in_ch if i == 0 else self.embed_dims[i - 1], out_ch=self.embed_dims[i], kernel_size=3)
            encoder[f'pool-{i + 1}'] = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleDict(encoder)

        self.bottle_neck = ConvBlock(in_ch=self.embed_dims[-2], out_ch=self.embed_dims[-1], kernel_size=3)

        self.embed_dims.reverse()
        decoder = OrderedDict()
        for i in range(len(self.embed_dims) - 1):
            decoder[f'up-{i + 1}'] = nn.ConvTranspose2d(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            decoder[f'expansive-{i + 1}'] = ConvBlock(in_ch=self.embed_dims[i], out_ch=self.embed_dims[i + 1], kernel_size=3)
        self.decoder = nn.ModuleDict(decoder)

        self.output = nn.Conv2d(in_channels=self.embed_dims[-1], out_channels=num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        residuals = []
        for i in range(len(self.embed_dims) - 1):
            x = self.encoder[f'contracting-{i + 1}'](x)
            residuals.append(x)
            x = self.encoder[f'pool-{i + 1}'](x)

        x = self.bottle_neck(x)

        residuals.reverse()
        for i in range(len(self.embed_dims) - 1):
            x = self.decoder[f'up-{i + 1}'](x)
            x = self.decoder[f'expansive-{i + 1}'](torch.cat((x, residuals[i]), dim=1))

        x = self.output(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=out_ch),
                                   nn.Conv2d(out_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=out_ch))

    def forward(self, x):
        return self.block(x)

