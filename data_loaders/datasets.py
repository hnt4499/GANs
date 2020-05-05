import os

import numpy as np
import pandas as pd
import torch

from base import BaseDataset


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing purposes.

    Parameters
    ----------
    num_samples : int
        Number of samples. Data will be a list of integers from 0 to
        `num_samples - 1`.

    """
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self._data = list(range(num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return idx

    @property
    def data(self):
        return self._data

    def sample(self, num_samples, random=False):
        """Sample a subset from the dataset.

        Parameters
        ----------
        num_samples : int
            Number of samples of the subset.
        shuffle : bool
            If True, sample the dataset randomly.
            Otherwise, return the subset of `num_samples` first entries of the
            dataset.

        """
        if num_samples > self.__len__():
            raise ValueError(
                "Number of samples ({}) exceeds the total number of "
                "images ({}).".format(num_samples, self.__len__()))
        if random:
            subset = np.random.choice(
                self.data, num_samples, replace=False).tolist()
        else:
            subset = list(range(num_samples))
        return subset


class ImageNetDataset(BaseDataset):
    """ImageNet dataset reader.

    Parameters
    ----------
    root : str
        Root directory of the ImageNet dataset, with the following tree
        structure:
            root
            ├── ILSVRC
            │   ├── Annotations
            │   ├── Data
            │   └── ImageSets
            ├── LOC_sample_submission.csv
            ├── LOC_synset_mapping.txt
            ├── LOC_train_solution.csv
            └── LOC_val_solution.csv
    cls : str
        A ImageNet class name, for example "n01531178", which corresponds to
        "goldfinch".
    transform : fn
        A function in `pre_processing.py` to be taken as the custom image
        transformation function.
    include_val : bool
        Whether to include images from validation set.

    """
    def __init__(self, root, cls, transform=None, include_val=False):
        # Initialize base class
        train_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/train/" + cls)
        super(ImageNetDataset, self).__init__(
            root=train_dir, transform=transform)

        # Get validation file names
        if include_val:
            val_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/val")
            val_df = pd.read_csv(os.path.join(root, "LOC_val_solution.csv"))

            # Get images containing the desired class
            def filter_fn(x):
                return cls in x
            filter = val_df["PredictionString"].apply(filter_fn)
            val_names = val_df["ImageId"][filter]
            # Get validation filepaths
            val_paths = [os.path.join(val_dir, val_name + ".JPEG")
                         for val_name in val_names]
            self.filepaths.extend(val_paths)


class CelebADataset(BaseDataset):
    """CelebA dataset reader.

    Parameters
    ----------
    root : str
        Root directory of the CelebA dataset, with the following tree
        structure:
            root
                ├── images [202599 entries]
                └── list_attr_celeba.txt
    transform : fn
        A function in `pre_processing.py` to be taken as the custom image
        transformation function.

    """
    def __init__(self, root, transform):
        # Initialize base class
        train_dir = os.path.join(root, "images")
        super(CelebADataset, self).__init__(
            root=train_dir, transform=transform)
