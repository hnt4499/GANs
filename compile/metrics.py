"""
All functions in this module must return another function that takes a trainer
and return the computed metric.
"""

import numpy as np
import torch

from base.base_metric import BaseMetric, FPMetric

from .metrics_utils import calculate_kid


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


class KIDScore(FPMetric):
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
    subset_size
    num_subsets
    name
    dp

    """
    def __init__(self, model, name, max_samples=None,
                 kid_batch_size=1024, dp=4):
        super(KIDScore, self).__init__(name, dp)
        # Feature extraction
        self.model = model
        self.max_samples = max_samples
        self.kid_batch_size = kid_batch_size
        # Latest std
        self.value_std = None

    def __str__(self):
        mean = self.repr.format(self.value)
        std = self.repr.format(self.value_std)
        return "{}: {}±{}".format(self.name, mean, std)

    def __call__(self, trainer):
        with torch.no_grad():
            # Parse trainer information if needed
            if self.value is None:
                self.init(trainer)

            # Generate fake samples. Keep all samples (real and generated)
            # on CPU for memory efficiency
            self.noise = torch.randn(
                self.num_samples, trainer.length_z, 1, 1,
                device=trainer.device)
            self.fake_samples = trainer.netG(self.noise).detach().cpu()

            # Extract features, use `id=0` for real samples because we need to
            # compute only once; use current batch index as `id` for fake
            # samples
            self.feats_real = self.model(
                self.real_samples, real=True, id=0)
            self.feats_fake = self.model(
                self.fake_samples, real=False,
                id=trainer.current_batch.batch_idx)

            # KID expects input to be in random order
            self.real_idxs = np.random.choice(
                np.arange(len(self.feats_real)), size=self.num_samples,
                replace=False)
            self.fake_idxs = np.random.choice(
                np.arange(len(self.feats_fake)), size=self.num_samples,
                replace=False)
            feats_real = self.feats_real[self.real_idxs]
            feats_fake = self.feats_fake[self.fake_idxs]

            # Compute KID from features extracted
            self.value, self.value_std = calculate_kid(
                feats_fake, feats_real, self.device,
                batch_size=self.kid_batch_size)

        return self.value

    def init(self, trainer):
        """Parse trainer information, only called once"""
        # Use the same device as trainer
        self.device = trainer.device
        # Get real samples and convert to torch.Tensor
        self.real_samples = trainer.data_loader.dataset.data
        self.real_samples = torch.stack(self.real_samples)
        # Compute number of samples
        if self.max_samples is None:
            self.num_samples = len(self.real_samples)
        else:
            self.num_samples = min(self.max_samples, len(self.real_samples))
        self.real_samples = self.real_samples[:self.num_samples]
