"""
All functions in this module must return another function that takes model
predictions and target labels, and return the computed loss.
"""

import torch
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


def ce_loss():
    def ce(output, target):
        """Classic cross entropy loss"""
        return F.cross_entropy(output, target)
    return ce


def conditional_shannon():
    def con_H(probs):
        """Expected value of Shannon entropy of conditional probabilities.
        Reference:
            Jost Tobias Springenberg. (2016). Unsupervised and Semi-supervised
            Learning with Categorical Generative Adversarial Networks
        """
        log_probs = torch.log(probs)
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
        log_probs = torch.log(p)
        return -(p * log_probs).sum()
    return mar_H
