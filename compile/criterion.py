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
    def nll(output, target, *args, **kwargs):
        return F.nll_loss(output, target)
    return nll


def bce_loss():
    def bce(output, target, *args, **kwargs):
        """Classic binary cross entropy loss."""
        return F.binary_cross_entropy(output, target)
    return bce


def ce_loss():
    def ce(output, target, *args, smooth_label=1, **kwargs):
        """Cross entropy loss with label smoothing.

        Parameters
        ----------
        smooth_label : float
            Smooth label for the discriminator.

        """
        if smooth_label == 1:
            return F.cross_entropy(output, target)
        else:
            # Generate one-hot encoding
            target_onehot = torch.zeros(
                *output.shape, dtype=output.dtype, device=target.device)
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            # Smooth labels
            target_onehot = target_onehot * smooth_label
            # Calculate loss
            log_softmax = F.log_softmax(output, dim=1)
            return -(log_softmax * target_onehot).sum(axis=1).mean(axis=0)
    return ce


"""
CatGAN-specific losses.
"""


def conditional_shannon():
    def con_H(probs, *args, **kwargs):
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
    def mar_H(probs, *args, **kwargs):
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
