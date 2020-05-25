"""
All functions in this module must return another function that takes model
predictions and target labels, and return the computed loss.
"""

import torch
import torch.nn as nn
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


"""
ImprovedGAN-specific losses.
"""


def log_sum_exp(x, dim=1):
    """Calculate LogSumExp (LSE) of a tensor:
        LSE(x) = log(∑e^x_i), where i = 1, 2, ..., `num_features`

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (b, f), where b is batch size and f is the number
        of features.
    dim : int
        Axis along which to compute LSE. (default: 1)

    Returns
    -------
    torch.Tensor
        Output tensor of shape (b,).

    """
    m = x.max(dim=dim)[0]  # torch.max() returns a tuple of two tensors
    s = (x - m.view(-1, 1)).exp().sum(dim=dim)
    return m + s.log()


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss.

    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.
    """
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()

    def forward(self, features_real, features_fake, **kwargs):
        real_shape = features_real.shape
        fake_shape = features_fake.shape
        # Check validity
        if real_shape != fake_shape:
            raise RuntimeError("Dimension mismatch: {} and {}".format(
                real_shape, fake_shape))
        if len(real_shape) not in [2, 4]:
            raise RuntimeError(
                "Invalid number of dimensions. Expected one of [2, 4], got {} "
                "instead.".format(len(real_shape)))
        # Apply adaptive average pooling if needed
        if len(real_shape) == 4:
            features_real = F.adaptive_avg_pool2d(
                features_real, output_size=(1, 1)).view(real_shape[0], -1)
            features_fake = F.adaptive_avg_pool2d(
                features_fake, output_size=(1, 1)).view(fake_shape[0], -1)
        features_real = features_real.mean(0)
        features_fake = features_fake.mean(0)
        return (features_real - features_fake).abs().mean()


class ImprovedGANDiscriminatorLoss:
    """
    Loss for ImprovedGAN discriminator. Any custom loss should inherit this
    base loss.

    Adapted from:
        https://github.com/Sleepychord/ImprovedGAN-pytorch

    """
    def __init__(self, unsupervised_weight=0.5):
        """Initialize ImprovedGAN discriminator loss.

        Parameters
        ----------
        unsupervised_weight : float or int
            Weight for unsupervised loss.

        """
        self.unsupervised_weight = unsupervised_weight

    def loss_real_with_labels(self, output_wl, labels):
        """Supervised loss for real data as described in the paper.
        Reference:
            Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A.,
            & Chen, X. (2016). Improved Techniques for Training GANs.

        Parameters
        ----------
        output_wl : torch.Tensor
            Discriminator outputs (without softmax) for data with labels.
        labels : type
            Ground truth.

        """
        logz_wl = log_sum_exp(output_wl)
        prob_wl = torch.gather(output_wl, 1, labels.view(-1, 1))
        loss_supervised = -prob_wl.mean() + logz_wl.mean()
        return loss_supervised

    def loss_real_without_labels(self, output_wol):
        """Unsupervised loss for real data as described in the paper.
        Reference:
            Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A.,
            & Chen, X. (2016). Improved Techniques for Training GANs.

        Parameters
        ----------
        output_wol : torch.Tensor
            Discriminator outputs (without softmax) for data without labels.

        """
        logz_wol = log_sum_exp(output_wol)
        loss_unsupervised_real = -logz_wol.mean() + F.softplus(logz_wol).mean()
        return loss_unsupervised_real

    def loss_fake(self, output):
        """Unsupervised loss for fake data as described in the paper.
        Reference:
            Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A.,
            & Chen, X. (2016). Improved Techniques for Training GANs.

        Parameters
        ----------
        output : torch.Tensor
            Discriminator outputs (without softmax) for generated (fake) data.

        """
        logz_fake = log_sum_exp(output)
        loss_unsupervised_fake = F.softplus(logz_fake).mean()
        return loss_unsupervised_fake


class ImprovedGANGeneratorLoss(FeatureMatchingLoss):
    pass
