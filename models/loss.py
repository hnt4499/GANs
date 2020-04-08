import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_loss(output, target):
    """Classic binary cross entropy loss."""
    return F.binary_cross_entropy(output, target)
