class BaseFeatureExtractor:
    """Base class for feature extractors. This class implements some protocols
    allowing sharing activations (features extracted by DL models) between
    different metrics.

    Attributes
    ----------
    model
        A PyTorch model for extracting features.
    id
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

    """
    def __init__(self):
        self.model = None  # model for extracting features

        self.id_real = None
        self.feats_real = None

        self.id_fake = None
        self.feats_fake = None

    def __call__(self, x, id, real=True):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
