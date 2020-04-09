from base import BaseDataLoader


class ImageNetLoader(BaseDataLoader):
    """ImageNet data loader.

    Parameters
    ----------
    dataset
        A dataset, instantiated object of a class defined in
        `data_loaders.datasets`.
    batch_size : int
        Mini-batch size for training.
    shuffle : bool
        Whether to shuffle the training set.
    num_workers : int
        The number of subprocesses to use for data loading. `0` means that the
        data will be loaded in the main process. (default: `0`)

    """
    def __init__(self, dataset, batch_size=128, shuffle=True, num_workers=1):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split=0.0, num_workers=num_workers)
