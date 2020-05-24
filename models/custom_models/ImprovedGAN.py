from abc import abstractmethod

import torch.nn as nn

from base import BaseGANDiscriminator
from .DCGAN import DCGANDiscriminator, DCGANGenerator
from utils.ops import View


class ImprovedGANDiscriminatorV0(BaseGANDiscriminator):
    """
    Base class for any version of ImprovedGAN discriminator.
    This differs significantly from the original implementation that this does
    not use Network-In-Network (NIN) layers. This is to keep the implementation
    simple since discriminator architecture should not affect the overall
    performance much.
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.
    """
    def __init__(self, optimizer, criterion, num_classes, image_size=32,
                 num_features=128, num_channels=3, conv_bias=False,
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
            (e.g, 64 -> 128 -> 256 -> 512 -> 1024).
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
        super(ImprovedGANDiscriminatorV0, self).__init__(
            optimizer, criterion, weights_init)

        # Cache data
        self.image_size = image_size
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv_bias = conv_bias
        self.negative_slope = negative_slope

        # Get layers from DCGAN, except the last layer
        self.dcgan = DCGANDiscriminator(
            image_size=image_size, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            negative_slope=negative_slope, optimizer=optimizer,
            criterion=criterion, weights_init=weights_init)
        self.num_layers = self.dcgan.num_layers
        # Build model
        self.build_model()
        # Remove DCGAN when done
        del self.dcgan
        # Initialize optimizer and model weights
        self.init()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError


class ImprovedGANGeneratorV0(DCGANGenerator):
    """
    Base class for any ImprovedGAN generator.
    Note that this implementation differs from the original implementation in
    the first layer, where transposed convolution operator is used, instead of
    using linear layer then reshaping as in the original implementation.
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.

    """
    def __init__(self, optimizer, criterion, image_size=32, input_length=100,
                 num_features=128, num_channels=3, conv_bias=False,
                 weights_init=None):
        super(ImprovedGANGeneratorV0, self).__init__(
            optimizer=optimizer, criterion=criterion, image_size=image_size,
            input_length=input_length, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            weights_init=weights_init)


class ImprovedGANDiscriminatorV1(ImprovedGANDiscriminatorV0):
    """ImprovedGAN discriminator with feature matching loss.
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.

    """
    def build_model(self):
        # Get layers from DCGAN, except the last layer
        self.copy_layers(self.dcgan, self.dcgan.modules[:-1])
        # The last convolutional layer will still double the number of
        # channels, as opposed to other models; but no Leaky ReLU and batchnorm
        # here.
        module_name = self.dcgan.modules[-1]
        out_channels = self.dcgan[-2].__getattr__("0").out_channels
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels * 2,
                kernel_size=4, stride=1, padding=0, bias=False),
            View("0", -1)
        )
        self[module_name] = module
        # The last layer is a linear layer
        self["Linear"] = nn.Linear(out_channels * 2, self.num_classes)


class ImprovedGANGeneratorV1(ImprovedGANGeneratorV0):
    """ImprovedGAN generator with feature matching loss.
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.

    """


