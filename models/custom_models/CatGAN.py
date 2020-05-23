import math

import torch.nn as nn

from base import BaseGANDiscriminator
from .DCGAN import DCGANDiscriminator, DCGANGenerator
from utils.ops import View


class CatGANDiscriminator(BaseGANDiscriminator):
    """CatGAN Discriminator. This differs from DCGAN that CatGAN discriminator
    is a multi-class classifier.
    Reference:
        Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
        Learning with Categorical Generative Adversarial Networks

    """
    def __init__(self, optimizer, criterion, num_classes, image_size=32,
                 num_features=64, num_channels=3, conv_bias=False,
                 negative_slope=0.2, weights_init=None):
        """
        Parameters
        ----------
        optimizer : fn
            A function initialized in `compile.optimizers` that takes only the
            model trainable parameters as input.
        criterion : fn
            A function initialized in `compile.criterion` that takes the model
            predictions and target labels, and return the computed loss.
        num_classes : int
            Number of classes for predictions. This should (but not
            necessarily) be equal to the number of classes in the dataset.
        image_size : int
            Discriminator's output image size. Must be of the form `2 ** n`,
            n >= 5 (e.g., 32 or 64).
        num_features : int
            Number of channels of the first layer. The number of channels of
            a layer is as twice as that of its precedent layer
            (e.g, 32 -> 64 -> 128 -> 256).
        num_channels : int
            One of [1, 3]. Number of input image channels.
        conv_bias : bool
            Whether to include bias in the convolutional layers.
        negative_slope : float
            Hypterparameter for the Leaky RELU layers.
        weights_init : fn
            A function initialized in `models.weights_init` that will then be
            passed to `model.apply()`. (default: None)

        """
        super(CatGANDiscriminator, self).__init__(
            optimizer, criterion, weights_init)

        # Cache data
        self.image_size = image_size
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv_bias = conv_bias
        self.negative_slope = negative_slope

        # Get layers from DCGAN, except the last layer
        dcgan = DCGANDiscriminator(
            image_size=image_size, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            negative_slope=negative_slope, optimizer=optimizer,
            criterion=criterion, weights_init=weights_init)
        self.num_layers = dcgan.num_layers
        self.copy_layers(dcgan, dcgan.modules[:-1])

        # Get information
        module_name = dcgan.modules[-1]
        out_channels = dcgan[-2].__getattr__("0").out_channels
        # Construct the last layer
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=num_classes,
                kernel_size=4, stride=1, padding=0, bias=False),
            View(-1, num_classes),
            nn.Softmax(dim=-1)
        )
        self[module_name] = module

        # Initialize optimizer and model weights
        self.init()


class CatGANGenerator(DCGANGenerator):
    """
    CatGAN Generator.
    Reference:
        Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
        Learning with Categorical Generative Adversarial Networks
    """
    def __init__(self, optimizer, criterion, image_size=32, input_length=100,
                 num_features=64, num_channels=3, conv_bias=False,
                 weights_init=None):
        super(CatGANGenerator, self).__init__(
            optimizer=optimizer, criterion=criterion, image_size=image_size,
            input_length=input_length, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            weights_init=weights_init)
