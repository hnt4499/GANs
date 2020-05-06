import os

import numpy as np
import pandas as pd
import torch

from base import BaseDataset, BaseDatasetNpy


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing purposes.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    fill : int or float or None
        If None, fill the dataset sequentially (i.e., as in `range`).
        If not None, fill the dataset with this value.
    shape : list of integers
        Sample's shape. For example, if None, each sample is a scalar.
    dtype
        Must be a valid numpy data type.
    """
    def __init__(self, num_samples, fill=None, shape=None, dtype="int"):
        # Cache data
        self.num_samples = num_samples
        self.fill = fill
        self.shape = shape
        self.dtype = dtype

        # Dataset shape
        data_shape = [num_samples]
        if shape is not None:
            data_shape += list(shape)
        # Create dummy dataset
        if fill is None:
            num_entries = np.prod(data_shape)
            self._data = np.arange(
                num_entries, dtype=dtype).reshape(data_shape)
        else:
            self._data = np.full(
                shape=data_shape, fill_value=fill, dtype=dtype)
        self._data = torch.from_numpy(self._data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._data[idx]

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
            idxs = np.random.choice(
                range(self.num_samples), num_samples, replace=False)
        else:
            idxs = list(range(num_samples))
        return self._data[idxs]


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


class CelebADatasetNpy(BaseDatasetNpy):
    """CelebA dataset reader from *npy file(s)."""
