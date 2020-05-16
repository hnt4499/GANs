import os

import numpy as np
import pandas as pd
import torch

from base import BaseDataset, BaseDatasetWithLabels, BaseDatasetNpy


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing purposes.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    fill : int or float or None
        If None, dataset is randomly sampled from a uniform distribution
        according to the `dtype` provided.
            If `dtype="int"` (or its variants), dataset is filled with random
            integers from 0 to 255 (inclusively).
            If `dtype="float" (or its variants)`, dataset is filled with random
            float point number between 0 (inclusively) and 1 (exclusively).
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
            if "int" in dtype:
                self._data = np.random.randint(
                    low=0, high=256, size=data_shape, dtype=dtype)
            else:
                self._data = np.random.rand(*data_shape).astype(dtype)
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


class CIFAR10DatasetWithLabels(BaseDatasetWithLabels):
    """CIFAR-10 dataset reader, which must previously be downloaded using
    `cifar2png` package: https://github.com/knjcode/cifar2png. This class's
    object, when called, will return one tensor of images and one tensor of
    their respective labels"""
    def __init__(self, root, transform, subset="train"):
        """Initialize CIFAR-10 dataset.

        Parameters
        ----------
        root : str
            Root directory of the CIFAR-10 dataset obtained from the package
            `cifar2png` with the following tree structure:
                root
                ├── train
                │   ├── airplane
                │   ├── automobile
                │   └── ...
                └── test
                    ├── airplane
                    ├── automobile
                    └── ...
        transform : fn
            A function in `pre_processing.py` to be taken as the custom image
            transformation function.
        subset : str
            Subset of data to use. One of ["train", "test", "both"]. If "both",
            combine and use both train and test set.

        """
        self.labels = list()
        self.cls = list()
        # Get classes and class mapping based on "train" folder
        self._get_cls(os.path.join(root, "train"))
        self._get_cls_mapping()

        if subset not in ["train", "test", "both"]:
            raise ValueError('Invalid subset. Expected one of ["train", '
                             '"test", "both"], got {} instead.'.format(subset))
        # Read both train and test data
        if subset == "both":
            r = list()
            for s in ["train", "test"]:
                self.root = os.path.join(root, s)
                self._get_file_paths()
                r.append(self.root)
            root = r  # a list containing paths to train and test directory
            self._get_file_paths = lambda x: None  # turn this off
        else:
            root = os.path.join(root, subset)
        # Initialize base class
        super(CIFAR10DatasetWithLabels, self).__init__(
            root=root, transform=transform)


class CIFAR10DatasetWithoutLabels(CIFAR10DatasetWithLabels):
    """CIFAR-10 dataset reader, which must previously be downloaded using
    `cifar2png` package: https://github.com/knjcode/cifar2png. This class's
    object, when called, will return only one tensor of images"""
    def __getitem__(self, idx):
        return super(CIFAR10DatasetWithoutLabels, self).__getitem__(idx)[0]
