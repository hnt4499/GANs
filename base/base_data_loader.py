from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders, where `sampler` is set to None by default.
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        """Initialize the data loader.

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
            The number of subprocesses to use for data loading. `0` means that
            the data will be loaded in the main process.

        Attributes
        ----------


        """
        self.n_samples = len(dataset)

        self.init_kwargs = {
            "dataset": dataset, "batch_size": batch_size,
            "shuffle": shuffle, "collate_fn": default_collate,
            "num_workers": num_workers
        }
        super().__init__(sampler=None, **self.init_kwargs)
