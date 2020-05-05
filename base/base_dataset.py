import os

from PIL import Image
import numpy as np
import torch
from torchvision import get_image_backend


class BaseDataset(torch.utils.data.Dataset):
    """Base class for any dataset implementation.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    transform : fn
        A function defined in `data_loaders.pre_processing.py` to be taken as
        the custom image transformation function.

    Attributes
    ----------
    filepaths : list
        List of all training file paths.
    root
    transform

    """
    def __init__(self, root, transform=None):
        # Get image file paths
        self.root = root
        self.filepaths = list()
        self._data = None  # store loaded images

        for filename in os.listdir(root):
            _, ext = os.path.splitext(filename)
            if ext.lower() not in IMG_EXTENSIONS:
                continue
            self.filepaths.append(os.path.join(root, filename))
        # Transformation function
        if transform is None:
            self.transform = lambda x: x  # identity function
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.transform(default_loader(self.filepaths[idx]))

    @property
    def data(self):
        """Return list of loaded images"""
        if self._data is None:
            self._data = list(map(self.__getitem__, range(self.__len__())))
        return self._data

    def sample(self, num_samples, random=False):
        """Sample a subset from the dataset.

        Parameters
        ----------
        num_samples : int
            Number of samples of the subset.
        random : bool
            If True, sample the dataset randomly.
            Otherwise, return the subset of `num_samples` first images of the
            dataset.

        """
        if num_samples > self.__len__():
            raise ValueError(
                "Number of samples ({}) exceeds the total number of "
                "images ({}).".format(num_samples, self.__len__()))
        if random:
            idxs = np.random.choice(
                np.arange(self.__len__()), num_samples, replace=False)
        else:
            idxs = np.arange(num_samples)
        fn = lambda idx: self[idx]
        return list(map(fn, idxs))


"""
Helper functions for data loading
Adapted from
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
"""

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """Default image reader."""
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
