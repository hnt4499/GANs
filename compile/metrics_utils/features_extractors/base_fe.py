import math

import torch
import torch.nn.functional as F


class BaseFeatureExtractor:
    """Base class for feature extractors. This class implements some protocols
    allowing sharing activations (features extracted by DL models) between
    different metrics.

    Parameters
    ----------
    model_initializer
        Feature extractor class.
    batch_size : int
        Mini-batch size for feature extractor.
    device : str
        String argument to pass to `torch.device`, for example, "cpu" or
        "cuda".
    init_kwargs
        These keyword arguments will be passed to `model_initializer` to
        initialize feature extraction model.

    Attributes
    ----------
    model
        A PyTorch model for extracting features.
    id_real, id_fake
        An unique ID determining current step of the training process. When
        calling this class instance with a different ID, features will be
        re-computed for the new ID. When calling this class instance with the
        same ID, cached features will be returned.
        This protocol allows sharing features between different metrics by
        using the same ID (for example, current batch index) at a step.
    feats_real : torch.Tensor
        Features extracted from real samples.
    feats_fake : torch.Tensor
        Features extracted from fake samples.
    model_initializer
    init_kwargs
    device

    """
    def __init__(self, model_initializer, batch_size, device="cpu",
                 **init_kwargs):
        self.model = None  # model for extracting features
        self.model_initializer = model_initializer
        self.init_kwargs = init_kwargs

        self.batch_size = batch_size
        self.device = torch.device(device=device)

        self.id_real = None
        self.feats_real = None

        self.id_fake = None
        self.feats_fake = None

    def __call__(self, x, id, real=True):
        """Base function for feed-forwarding input. Overwrite if needed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        id
            An unique ID with respect to the type of input (real of fake)
            determining current step of the training process.
        real : bool
            Whether the inputs are real samples.

        """
        # Initialize model if needed
        if self.model is None:
            self.model = self.model_initializer(**self.init_kwargs)
            self.model = self.model.to(device=self.device)
        # Real samples
        if real:
            # Compute new batch of inputs
            if self.id_real is None or self.id_real != id:
                self.id_real = id  # update current ID
                self.feats_real = self._forward(x)  # update real features
            return self.feats_real
        # Fake samples
        else:
            # Compute new batch of inputs
            if self.id_fake is None or self.id_fake != id:
                self.id_fake = id  # update current ID
                self.feats_fake = self._forward(x)  # update fake features
            return self.feats_fake

    def _forward(self, x):
        """Helper function to forward batched inputs"""
        num_samples = x.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)
        feats = list()

        self.model.eval()  # important: don't replace this with torch.no_grad()
        for batch in range(num_batches):
            start = batch * self.batch_size
            end = start + self.batch_size

            feat = self.model(x[start:end].to(device=self.device))
            # If model output is not scalar, apply global spatial average
            # pooling. This happens if you choose too shallow layer.
            shape = list(feat.shape)
            if any(s != 1 for s in shape[2:]):
                feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))
            # Flatten feature(s)
            feat = feat.cpu().view(shape[0], -1)
            feats.append(feat)

        return torch.cat(feats, dim=0)

    def __eq__(self, other):
        """Base function for comparing this class instance with another object.
        Overwrite if needed.

        Parameters
        ----------
        other
            Python object to compare to.

        """
        # Simply compare all initializer's keyword arguments
        for this_kwarg, this_value in self.init_kwargs.items():
            other_value = other.init_kwargs.get(this_kwarg)
            if other_value != this_value:
                return False
        return True
