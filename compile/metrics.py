"""
All functions in this module must return another function that takes a trainer
and return the computed metric.
"""

import numpy as np
import torch

from base.base_metric import FPMetric, FEMetric

from .metrics_utils import calculate_kid, calculate_fid


class Accuracy(FPMetric):
    """Calculate accuracy of the discriminator classifying generated images.

    Parameters
    ----------
    name : str
        Name of the metric, for example "acc_d".
    dp : int
        Number of decimal places to be printed. (default: 4)

    Attributes
    ----------
    value
        The latest computed metric value.
    name
    dp

    """
    def __init__(self, name, dp=4):
        super(Accuracy, self).__init__(name, dp)

    def __call__(self, trainer):
        with torch.no_grad():
            pred = trainer.current_batch.output_D
            pred = (pred >= 0.5).type(torch.float32)
            # Get target labels
            target = torch.full(
                (trainer.current_batch.batch_size,),
                trainer.fake_label, device=trainer.device)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
        self.value = correct / len(target)
        return self.value


class KIDScore(FEMetric):
    """Calculate the KID score.

    Parameters
    ----------
    model
        An instantiated model defined in
        `compile.metrics_utils.features_extractors`, which returns the
        extracted features when called.
    name : str
        Name of the metric, for example "kid".
    max_samples : int
        Maximum number of samples to use. If None, use all samples.
    kid_batch_size : int
        Batch size for KID calculation.
    dp : int
        Number of decimal places to be printed. (default: 4)

    Attributes
    ----------
    value
        The latest computed KID score (mean).
    value_std
        The latest computed KID score (std).
    max_samples
    batch_size
    name
    dp

    """
    def __init__(self, model, name, max_samples=None,
                 kid_batch_size=1024, dp=4):
        super(KIDScore, self).__init__(
            model=model, name=name, max_samples=max_samples, dp=dp)
        self.kid_batch_size = kid_batch_size
        self.value_std = None  # latest std

    def __str__(self):
        mean = self.repr.format(self.value)
        std = self.repr.format(self.value_std)
        return "{}: {}±{}".format(self.name, mean, std)

    def _calc_score(self, feats_real, feats_fake):
        self.value, self.value_std = calculate_kid(
            feats_fake, feats_real, device=self.device,
            batch_size=self.kid_batch_size)
        return self.value


class FIDScore(FEMetric):
    """Calculate the FID score.

    Parameters
    ----------
    model
        An instantiated model defined in
        `compile.metrics_utils.features_extractors`, which returns the
        extracted features when called.
    name : str
        Name of the metric, for example "fid".
    max_samples : int
        Maximum number of samples to use. If None, use all samples.
    dp : int
        Number of decimal places to be printed. (default: 4)

    Attributes
    ----------
    value
        The latest computed FID score.
    max_samples
    name
    dp

    """
    def __init__(self, model, name, max_samples=None, dp=4):
        super(FIDScore, self).__init__(
            model=model, name=name, max_samples=max_samples, dp=dp)

    def _calc_score(self, feats_real, feats_fake):
        self.value = calculate_fid(
            feats_fake, feats_real, device=self.device)
        return self.value
