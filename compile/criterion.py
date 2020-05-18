"""
All functions in this module must return another function that takes model
predictions and target labels, and return the computed loss.
"""

import torch
import torch.nn.functional as F

from utils import Cache


"""
General losses.
"""


def nll_loss():
    def nll(output, target):
        return F.nll_loss(output, target)
    return nll


def bce_loss():
    def bce(output, target):
        """Classic binary cross entropy loss."""
        return F.binary_cross_entropy(output, target)
    return bce


def ce_loss():
    def ce(output, target):
        """Classic cross entropy loss"""
        return F.cross_entropy(output, target)
    return ce


"""
CatGAN-specific losses.
"""


def conditional_shannon():
    def con_H(probs):
        """Expected value of Shannon entropy of conditional probabilities.
        Reference:
            Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
            Learning with Categorical Generative Adversarial Networks
        """
        log_probs = torch.log(probs + 1e-6)  # for numerical stability
        p = (-probs * log_probs).sum(axis=1)
        return p.mean()
    return con_H


def marginal_shannon():
    def mar_H(probs):
        """Estimate of Shannon entropy of marginal probabilities.
        Reference:
            Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
            Learning with Categorical Generative Adversarial Networks
        """
        p = probs.mean(axis=0)
        log_probs = torch.log(p + 1e-6)  # for numerical stability
        return -(p * log_probs).sum()
    return mar_H


class CatGANDiscriminatorLoss(Cache):
    """Loss for CatGAN discriminator"""
    def __init__(self, con_loss, mar_loss, ce_loss, lambd=1):
        """Initialize CatGAN discriminator loss.
        Reference:
            Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
            Learning with Categorical Generative Adversarial Networks

        Parameters
        ----------
        con_loss : fn
            Conditional loss. One of the conditional losses for CatGAN defined
            in `compile/criterion.py`.
        mar_loss : fn
            Marginalized loss. One of the marginalized losses for CatGAN
            defined in `compile/criterion.py`.
        ce_loss : fn
            Cross-entropy loss.
        lambd : float
            Weighting factor of the cross-entropy loss.

        """
        super(CatGANDiscriminatorLoss, self).__init__(
            con_loss=con_loss, mar_loss=mar_loss, ce_loss=ce_loss, lambd=lambd)


class CatGANGeneratorLoss(Cache):
    """Loss for CatGAN generator"""
    def __init__(self, con_loss, mar_loss):
        """Initialize CatGAN generator loss.
        Reference:
            Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
            Learning with Categorical Generative Adversarial Networks

        Parameters
        ----------
        con_loss : fn
            Conditional loss. One of the conditional losses for CatGAN defined
            in `compile/criterion.py`.
        mar_loss : fn
            Marginalized loss. One of the marginalized losses for CatGAN
            defined in `compile/criterion.py`.

        """
        super(CatGANGeneratorLoss, self).__init__(
            con_loss=con_loss, mar_loss=mar_loss)
