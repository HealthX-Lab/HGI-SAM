import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, embed_dims):
        """
        A 2D UNet class

        :param in_ch: number of channels in input image
        :param num_classes: number of classes
        :param embed_dims: list of embedding dimensions
        """
        super().__init__()

        self.embed_dims = embed_dims
        self.num_classes = num_classes

        self.encoder = nn.ModuleDict()
        for i in range(len(self.embed_dims) - 1):
            self.encoder[f'contracting-{i + 1}'] = ConvBlock(in_ch=in_ch if i == 0 else self.embed_dims[i - 1], out_ch=self.embed_dims[i], kernel_size=3)
            self.encoder[f'pool-{i + 1}'] = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottle_neck = ConvBlock(in_ch=self.embed_dims[-2], out_ch=self.embed_dims[-1], kernel_size=3)

        self.embed_dims.reverse()
        self.decoder = nn.ModuleDict()
        for i in range(len(self.embed_dims) - 1):
            self.decoder[f'up-{i + 1}'] = nn.ConvTranspose2d(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.decoder[f'expansive-{i + 1}'] = ConvBlock(in_ch=2 * self.embed_dims[i + 1], out_ch=self.embed_dims[i + 1], kernel_size=3)

        # we use a convolutional head which turned out to be more efficient
        self.head = nn.Conv2d(in_channels=self.embed_dims[-1], out_channels=num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        # passing input through encoder layers and keeping the residuals
        residuals = []
        for i in range(len(self.embed_dims) - 1):
            x = self.encoder[f'contracting-{i + 1}'](x)
            residuals.append(x)
            x = self.encoder[f'pool-{i + 1}'](x)

        # passing the encoder output through bottleneck
        x = self.bottle_neck(x)

        # passing the concatenation of residuals and encodings through decoder
        residuals.reverse()
        for i in range(len(self.embed_dims) - 1):
            x = self.decoder[f'up-{i + 1}'](x)
            x = self.decoder[f'expansive-{i + 1}'](torch.cat((x, residuals[i]), dim=1))

        # do the prediction
        x = self.head(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        A helper class for UNet model which is composed of 2 consecutive convolutional layers with ReLU activation

        :param in_ch: number of input channels
        :param out_ch: number of output channels
        :param kernel_size: kernel size of convolutional layers
        """
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

