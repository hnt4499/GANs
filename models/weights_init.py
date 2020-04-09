"""
All functions in this module must return another function that will then be
passed to `model.apply()`.
"""

import torch.nn as nn


def DCGAN_wi(conv_mean=0.0, conv_std=0.02, bn_mean=1.0, bn_std=0.02):
    """Default DCGAN weights initializer wrapper, which returns a function to
    be passed to `model.apply()`.
    Adapted from
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L113

    Parameters
    ----------
    conv_mean : float
        Mean of normal distribution for convolutional layers.
    conv_std : type
        Standard deviation of normal distribution for convolutional layers.
    bn_mean : type
        Mean of normal distribution for batch normalization layers.
    bn_std : type
        Standard deviation of normal distribution for batch normalization
        layers.

    Returns
    -------
    DCGAN_weights_init
        A function to be passed to `model.apply()`.

    """
    def DCGAN_weights_init(m):
        """Default DCGAN weights initializer for all convolutional,
        fractionally-strided convolutional and batch normalization layers using
        a normal distribution.
        Adapted from
            https://github.com/pytorch/examples/blob/master/dcgan/main.py#L113
        """
        class_name = m.__class__.__name__
        if "Conv" in class_name:
            nn.init.normal_(m.weight.data, conv_mean, conv_std)
        elif "BatchNorm" in class_name:
            nn.init.normal_(m.weight.data, bn_mean, bn_std)
            nn.init.constant_(m.bias.data, 0)
    return DCGAN_weights_init
