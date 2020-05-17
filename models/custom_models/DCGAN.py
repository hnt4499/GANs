import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from base import BaseGANDiscriminator, BaseGANGenerator


class DCGANDiscriminator(BaseGANDiscriminator):
    """DCGAN Discriminator. Note that all layers are channel-first.
    Adapted from:
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L164

    Parameters
    ----------
    image_size : int
        Discriminator's output image size. Must be of the form `2 ** n`, n >= 6
        (e.g., 64 or 128).
    num_features : int
        Number of channels of the first layer. The number of channels of
        a layer is as twice as that of its precedent layer
        (e.g, 64 -> 128 -> 256 -> 512 -> 1024).
    num_channels : int
        One of [1, 3]. Number of input image channels.
    conv_bias : bool
        Whether to include bias in the convolutional layers.
    negative_slope : float
        Hypterparameter for the Leaky RELU layers.
    optimizer : fn
        A function initialized in `compile.optimizers` that takes only the
        model trainable parameters as input.
    criterion : fn
        A function initialized in `compile.criterion` that takes the model
        predictions and target labels, and return the computed loss.
    weights_init : fn
        A function initialized in `models.weights_init` that will then be
        passed to `model.apply()`. (default: None)

    """
    def __init__(self, image_size, num_features, num_channels, conv_bias,
                 negative_slope, optimizer, criterion, weights_init=None):

        super(DCGANDiscriminator, self).__init__(
            optimizer, criterion, weights_init)

        # Cache data
        self.image_size = image_size
        self.num_features = num_features
        self.num_channels = num_channels
        self.conv_bias = conv_bias
        self.negative_slope = negative_slope
        # Number of intermediate layers
        num_layers = math.log(image_size, 2) - 2
        if num_layers.is_integer() and num_layers >= 4:
            self.num_layers = int(num_layers)
        else:
            raise ValueError(
                "Invalid value for `image_size`: {}".format(image_size))
        # Number of channels
        if num_channels not in [1, 3]:
            raise ValueError("Invalid value for `num_channels`. Expected one "
                             "of [1, 3], got {} instead.".format(num_features))

        seq = OrderedDict()
        li = 0  # convolutional layer index
        layer_name = "Conv2d_{}_4x4".format(li)
        # First layer
        seq[layer_name] = nn.Sequential(*[
            nn.Conv2d(
                in_channels=num_channels, out_channels=num_features,
                kernel_size=4, stride=2, padding=1, bias=conv_bias),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])
        # The rest of the intermediate layers
        out_channels = num_features
        for i in range(self.num_layers - 1):
            # Layer name
            li += 1
            layer_name = "Conv2d_{}_4x4".format(li)

            in_channels = out_channels
            out_channels = in_channels * 2
            seq[layer_name] = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope, inplace=True)
            ])
        # The output layer
        li += 1
        layer_name = "Conv2d_{}_4x4".format(li)
        seq[layer_name] = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels, out_channels=1, kernel_size=4,
                stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ])

        # Construct using OrderedDict
        self.construct(seq)
        # Initialize optimizer and model weights
        self.init()


class DCGANGenerator(BaseGANGenerator):
    """DCGAN Generator. Note that all layers are channel-first.
    Adapted from:
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L122

    Parameters
    ----------
    image_size : int
        Generator's output image size. Must be of the form `2 ** n`, n >= 6
        (e.g., 64 or 128).
    input_length : int
        Length of the noise input `z`. `z` is simply "transposed z-dimensional
        vector" of shape (1, 1, z).
    num_features : int
        Number of channels of the second-last layer (where the last layer is
        the output fake image). The number of channels of a layer is as twice
        as that of its successive layer (e.g, 1024 -> 512 -> 256 -> 128 -> 64).
    num_channels : int
        Number of output image channels.
    conv_bias : bool
        Whether to include bias in the fractionally-strided convolutional
        layers.
    optimizer : fn
        A function initialized in `compile.optimizers` that takes only the
        model trainable parameters as input.
    criterion : fn
        A function initialized in `compile.criterion` that takes the model
        predictions and target labels, and return the computed loss.
    weights_init : fn
        A function initialized in `models.weights_init` that will then be
        passed to `model.apply()`. (default: None)

    """
    def __init__(self, image_size, input_length, num_features, num_channels,
                 conv_bias, optimizer, criterion, weights_init=None):

        super(DCGANGenerator, self).__init__(
            optimizer, criterion, weights_init, output_range=(-1.0, 1.0))

        # Cache data
        self.image_size = image_size
        self.input_length = input_length
        self.num_features = num_features
        self.num_channels = num_channels
        self.conv_bias = conv_bias
        # Number of intermediate layers
        num_layers = math.log(image_size, 2) - 2
        if num_layers.is_integer() and num_layers >= 4:
            self.num_layers = int(num_layers)
        else:
            raise ValueError(
                "Invalid value for `image_size`: {}".format(image_size))
        # Number of channels
        if num_channels not in [1, 3]:
            raise ValueError("Invalid value for `num_channels`. Expected one "
                             "of [1, 3], got {} instead.".format(num_features))

        seq = list()
        # First layer
        out_channels = num_features * (2 ** (self.num_layers - 1))
        seq.extend([
            nn.ConvTranspose2d(
                in_channels=input_length, out_channels=out_channels,
                kernel_size=4, stride=1, padding=0, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        # The rest of the intermediate layers
        for i in range(self.num_layers - 1):
            in_channels = out_channels
            out_channels = num_features * (2 ** (self.num_layers - 2 - i))
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
                in_channels=out_channels, out_channels=num_channels,
                kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ])

        # Forward sequence
        self.fw = nn.Sequential(*seq)
        # Initialize optimizer and model weights
        self.init()

    def forward(self, x):
        return self.fw(x)
