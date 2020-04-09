"""
All functions in this module must return another function that takes model
predictions and target labels, and return the computed loss.
"""

import torch.nn.functional as F


def nll_loss():
    def nll(output, target):
        return F.nll_loss(output, target)
    return nll


def bce_loss():
    def bce(output, target):
        """Classic binary cross entropy loss."""
        return F.binary_cross_entropy(output, target)
    return bce
