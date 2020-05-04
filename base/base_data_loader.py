import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            "dataset": dataset, "batch_size": batch_size,
            "shuffle": self.shuffle, "collate_fn": default_collate,
            "num_workers": num_workers
        }
        super().__init__(sampler=None, **self.init_kwargs)
