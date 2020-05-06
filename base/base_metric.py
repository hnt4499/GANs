import numpy as np
import torch

from utils import forward_batch


class BaseMetric:
    """Base class for any metric implementation. Subclass must implement
    `__str__`, which is used to print useful logs, and `__call__`, which is
    used at the end of every epoch to compute the metric.

    Parameters
    ----------
    name : str
        Name of the metric, for example "acc_D".

    Attributes
    ----------
    value
        The latest computed metric value.
    name

    """
    def __init__(self, name):
        self.name = name
        self.value = None  # store latest computed value

    def __str__(self):
        raise NotImplementedError

    def __call__(self, trainer):
        raise NotImplementedError


class FPMetric(BaseMetric):
    """Base class for metrics whose value is floating point number. This base
    class implements default behavior for __str__.

    Parameters
    ----------
    name : str
        Name of the metric, for example "acc_D".
    dp : int
        Number of decimal places to be printed.

    Attributes
    ----------
    value
        The latest computed metric value.
    name
    dp

    """
    def __init__(self, name, dp=4):
        super(FPMetric, self).__init__(name)
        self.dp = dp
        self.repr = "{" + ":.{}f".format(dp) + "}"

    def __str__(self):
        # Format `self.value` to `dp` decimal places
        num = self.repr.format(self.value)
        return "{}: {}".format(self.name, num)


class FEMetric(FPMetric):
    """Base class for metrics which use features (activations) extracted from
    a DL model, for example, KID using Inception v3.

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
    dp : int
        Number of decimal places to be printed. (default: 4)

    """
    def __init__(self, model, name, max_samples=None, dp=4):
        super(FEMetric, self).__init__(name, dp=dp)

        self.model = model
        self.max_samples = max_samples

    def _get_device(self):
        self.device = self.trainer.device

    def _get_real_samples(self):
        """Helper function to get real samples without shuffling"""
        # Get the whole dataset
        if self.max_samples is None:
            if not hasattr(self, "real_samples"):
                self.real_samples = self.trainer.data_loader.dataset.data
                self.num_samples = len(self.real_samples)
        # Get a random subset if `max_samples` is specified
        # Re-sample every time this function gets called
        else:
            self.real_samples = self.trainer.data_loader.dataset.sample(
                num_samples=self.max_samples, random=True)
            self.num_samples = self.max_samples
        if isinstance(self.real_samples, list):
            self.real_samples = torch.stack(self.real_samples)
        return self.real_samples

    def _get_fake_samples(self):
        """Helper function to get fake samples without shuffling"""
        # Generate random noise then fake samples
        self.noise = torch.randn(
            self.num_samples, self.trainer.length_z, 1, 1)
        self.fake_samples = forward_batch(
            model=self.trainer.netG, inputs=self.noise,
            device=self.trainer.device,
            batch_size=self.trainer.data_loader.batch_size)
        return self.fake_samples

    def _get_features(self, real_samples, fake_samples):
        """Helper function to extract features from real and fake samples. This
        function is designed to get called before `_get_shuffled_feats` since
        we want to share the extracted features among different metrics using
        the same feature extractor, but we don't want their randomly chosen
        features to be identical."""
        # Use `id=0` for real samples because we need to compute only once;
        # use current batch index as `id` for fake samples
        self.feats_real = self.model(
            self.trainer, real_samples, real=True, id=0)
        self.feats_fake = self.model(
            self.trainer, fake_samples, real=False,
            id=self.trainer.current_batch.batch_idx)
        return self.feats_real, self.feats_fake

    def _get_shuffled_feats(self, feats_real, feats_fake):
        """Helper function to get randomly shuffled real and fake features,
        since this type of metric expects input to be in random order"""
        self.real_idxs = np.random.choice(
            np.arange(len(feats_real)), size=self.num_samples, replace=False)
        self.fake_idxs = np.random.choice(
            np.arange(len(feats_fake)), size=self.num_samples, replace=False)
        feats_real_shuffled = feats_real[self.real_idxs]
        feats_fake_shuffled = feats_fake[self.fake_idxs]
        return feats_real_shuffled, feats_fake_shuffled

    def _calc_score(self, feats_real, feats_fake):
        """Helper function to calculate score from extracted features"""
        raise NotImplementedError

    def __call__(self, trainer):
        with torch.no_grad():
            # Cache trainer for future use
            self.trainer = trainer
            # Get device
            self._get_device()
            # Get real samples
            real_samples = self._get_real_samples()
            # Get fake samples
            fake_samples = self._get_fake_samples()
            # Extract features
            feats_real, feats_fake = self._get_features(
                real_samples, fake_samples)
            # Choose a randomly shuffled subset
            feats_real, feats_fake = self._get_shuffled_feats(
                feats_real, feats_fake)
            # Compute score from extracted features
            score = self._calc_score(feats_real, feats_fake)

        return score
