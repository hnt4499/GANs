import math

import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from . import weights_init as wi


class Discriminator(BaseModel):
    """DCGAN Discriminator. Note that all layers are channel-first.
    Adapted from:
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L164

    Parameters
    ----------
    image_size : int
        Discriminator's output image size. Must be of the form `2 ** n`, n >= 6
        (e.g., 64 or 128).
    ndf : int
        Number of channels of the first layer. The number of channels of
        a layer is as twice as that of its precedent layer
        (e.g, 64 -> 128 -> 256 -> 512 -> 1024).
    nc : int
        Number of input image channels.
    conv_bias : bool
        Whether to include bias in the convolutional layers.
    negative_slope : float
        Hypterparameter for the Leaky RELU layers.
    weights_init : str
        A string, name of the function in `weights_init.py` to be taken as the
        custom weight initialization function. (default: "DCGAN_wi")

    """
    def __init__(self, image_size=64, ndf=64, nc=3, conv_bias=False,
                 negative_slope=0.2, weights_init="DCGAN_wi"):
        super(Discriminator, self).__init__()
        # Get weights initializer
        self.weights_init = getattr(wi, weights_init)
        # Number of intermediate layers
        n_layers = math.log(image_size, 2) - 2
        if n_layers.is_integer() and n_layers >= 4:
            self.n_layers = int(n_layers)
        else:
            raise ValueError(
                "Invalid value for `image_size`: {}".format(image_size))

        seq = list()
        # First layer
        seq.extend([
            nn.Conv2d(
                in_channels=nc, out_channels=ndf, kernel_size=4, stride=2,
                padding=1, bias=conv_bias),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])
        # The rest of the intermediate layers
        out_channels = ndf
        for i in range(self.n_layers - 1):
            in_channels = out_channels
            out_channels = in_channels * 2
            seq.extend([
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope, inplace=True)
            ])
        # The output layer
        seq.extend([
            nn.Conv2d(
                in_channels=out_channels, out_channels=1, kernel_size=4,
                stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ])
        # Forward sequence
        self.fw = nn.Sequential(*seq)
        # Initialize weights
        self.apply(self.weights_init)

    def forward(self, x):
        return self.fw(x)


class Generator(BaseModel):
    """DCGAN Generator. Note that all layers are channel-first.
    Adapted from:
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L122

    Parameters
    ----------
    image_size : int
        Generator's output image size. Must be of the form `2 ** n`, n >= 6
        (e.g., 64 or 128).
    nz : int
        Length of the noise input `z`. `z` is simply "transposed z-dimensional
        vector" of shape (1, 1, z).
    ngf : int
        Number of channels of the second-last layer (where the last layer is
        the output fake image). The number of channels of a layer is as twice
        as that of its successive layer (e.g, 1024 -> 512 -> 256 -> 128 -> 64).
    nc : int
        Number of output image channels.
    conv_bias : bool
        Whether to include bias in the fractionally-strided convolutional
        layers.
    weights_init : str
        A string, name of the function in `weights_init.py` to be taken as the
        custom weight initialization function. (default: "DCGAN_wi")

    """
    def __init__(self, image_size=64, nz=100, ngf=64, nc=3, conv_bias=False,
                 weights_init="DCGAN_wi"):
        super(Generator, self).__init__()
        # Get weights initializer
        self.weights_init = getattr(wi, weights_init)
        # Number of intermediate layers
        n_layers = math.log(image_size, 2) - 2
        if n_layers.is_integer() and n_layers >= 4:
            self.n_layers = int(n_layers)
        else:
            raise ValueError(
                "Invalid value for `image_size`: {}".format(image_size))

        seq = list()
        # First layer
        out_channels = ngf * (2 ** (self.n_layers - 1))
        seq.extend([
            nn.ConvTranspose2d(
                in_channels=nz, out_channels=out_channels, kernel_size=4,
                stride=1, padding=0, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        # The rest of the intermediate layers
        for i in range(self.n_layers - 1):
            in_channels = out_channels
            out_channels = ngf * (2 ** (self.n_layers - 2 - i))
            seq.extend([
                nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=4, stride=2, padding=1, bias=conv_bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        # The output layer
        seq.extend([
            nn.ConvTranspose2d(
                in_channels=out_channels, out_channels=nc, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.Tanh()
        ])
        # Forward sequence
        self.fw = nn.Sequential(*seq)
        # Initialize weights
        self.apply(self.weights_init)

    def forward(self, x):
        return self.fw(x)


class DCGAN:
    """DCGAN Model. Note that all layers in discriminator and generator are
    channel-first.

    Parameters
    ----------
    image_size : int
        Discriminator's input and generator's output image size. Must be of the
        form `2 ** n`, n >= 6 (e.g., 64 or 128).
    ndf : int
        Number of channels of the first layer in the discriminator. The number
        of channels of a layer is as twice as that of its precedent layer
        (e.g, 64 -> 128 -> 256 -> 512 -> 1024).
    nz : int
        Length of the noise input `z` of the generator. `z` is simply
        "transposed z-dimensional vector" of shape (1, 1, z).
    ngf : int
        Number of channels of the second-last layer in the generator (where the
        last layer is the output fake image). The number of channels of a layer
        is as twice as that of its successive layer
        (e.g, 1024 -> 512 -> 256 -> 128 -> 64).
    nc : int
        Number of image channels.
    conv_bias : bool
        Whether to include bias in the convolutional layers.
    negative_slope : float
        Hypterparameter for the Leaky RELU layers of the discriminator.
    weights_init : str
        A string, name of the function in `weights_init.py` to be taken as the
        custom weight initialization function. (default: "DCGAN_wi")

    Attributes
    ----------
    netD
        Discriminator
    netG
        Generator

    """
    def __init__(self, image_size=64, ndf=64, nz=100, ngf=64, nc=3,
                 conv_bias=False, negative_slope=0.2, weights_init="DCGAN_wi"):
        # Discriminator
        self.netD = Discriminator(
            image_size=image_size, ndf=ndf, nc=nc, conv_bias=conv_bias,
            negative_slope=negative_slope, weights_init=weights_init)
        # Generator
        self.netG = Generator(
            image_size=image_size, nz=nz, ngf=ngf, nc=nc, conv_bias=conv_bias,
            weights_init=weights_init)
