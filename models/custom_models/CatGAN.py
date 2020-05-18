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
    def __init__(self, image_size, num_features, num_channels, num_classes,
                 conv_bias, negative_slope, optimizer, criterion,
                 weights_init=None):
        """
        Parameters
        ----------
        image_size : int
            Discriminator's output image size. Must be of the form `2 ** n`,
            n >= 6 (e.g., 64 or 128).
        num_features : int
            Number of channels of the first layer. The number of channels of
            a layer is as twice as that of its precedent layer
            (e.g, 64 -> 128 -> 256 -> 512 -> 1024).
        num_channels : int
            One of [1, 3]. Number of input image channels.
        num_classes : int
            Number of classes for predictions. This should (but not
            necessarily) be equal to the number of classes in the dataset.
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
        super(CatGANDiscriminator, self).__init__(
            optimizer, criterion, weights_init)

        # Cache data
        self.image_size = image_size
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_classes = num_classes
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

        # Get layers from DCGAN, except the last layer
        dcgan = DCGANDiscriminator(
            image_size=image_size, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            negative_slope=negative_slope, optimizer=optimizer,
            criterion=criterion, weights_init=weights_init)
        for module_name in dcgan.modules[:-1]:
            self[module_name] = dcgan[module_name]

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
