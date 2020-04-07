import torch.nn as nn


def DCGAN_wi(m):
    """Default DCGAN weights initializer for all convolutional,
    fractionally-strided convolutional and batch normalization layers.
    Adapted from
        https://github.com/pytorch/examples/blob/master/dcgan/main.py#L113"""
    class_name = m.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
