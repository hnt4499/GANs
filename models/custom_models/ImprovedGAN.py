import torch.nn as nn

from base import BaseGANDiscriminator
from .DCGAN import DCGANDiscriminator, DCGANGenerator
from utils.ops import View


class ImprovedGANDiscriminator(BaseGANDiscriminator):
    """ImprovedGAN Discriminator. This differs significantly from the original
    implementation that this does not use Network-In-Network (NIN) layers. This
    is to keep the implementation simple since discriminator architecture
    should not affect the overall performance much.
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.

    """
    def __init__(self, optimizer, criterion, num_classes, image_size=32,
                 num_features=128, num_channels=3, conv_bias=False,
                 negative_slope=0.2, weights_init=None):
        super(ImprovedGANDiscriminator, self).__init__(
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

        # The last convolutional layer will still double the number of
        # channels, as opposed to other models; but no Leaky ReLU here.
        module_name = dcgan.modules[-1]
        out_channels = dcgan[-2].__getattr__("0").out_channels
        module = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels * 2,
                kernel_size=4, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels * 2),
            View("0", -1)
        )
        self[module_name] = module
        # The last layer is a linear layer
        self["Linear"] = nn.Linear(out_channels * 2, num_classes)
        # Initialize optimizer and model weights
        self.init()


class ImprovedGANGenerator(DCGANGenerator):
    """
    ImprovedGAN Generator.
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
        super(ImprovedGANGenerator, self).__init__(
            optimizer=optimizer, criterion=criterion, image_size=image_size,
            input_length=input_length, num_features=num_features,
            num_channels=num_channels, conv_bias=conv_bias,
            weights_init=weights_init)
