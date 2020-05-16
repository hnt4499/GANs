import os

from PIL import Image
import numpy as np
import torch
from torchvision import get_image_backend

from utils import read_json


class BaseDataset(torch.utils.data.Dataset):
    """Base class for any dataset implementation which reads data from an image
    folder. This base class should only be used for small datasets (< 10K). For
    large datasets, see `BaseDatasetNpy`, which reads data from numpy array of
    pre-processed images.

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
        self._get_file_paths()
        # Transformation function
        if transform is None:
            self.transform = lambda x: x  # identity function
        else:
            self.transform = transform

    def _get_file_paths(self):
        """Get training file paths with the depth of one. Subclasses may
        overwrite this function if needed"""
        for filename in os.listdir(self.root):
            _, ext = os.path.splitext(filename)
            if ext.lower() not in IMG_EXTENSIONS:
                continue
            self.filepaths.append(os.path.join(self.root, filename))

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


class BaseDatasetWithLabels(BaseDataset):
    """Base class for dataset implementation which reads data from an image
    folder with class information.
    This base class should only be used for small datasets (< 10K). For
    large datasets, see `BaseDatasetNpy`, which reads data from numpy array of
    pre-processed images.

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
    def _get_cls(self, root):
        """Get class names. Subclasses may overwrite this function if needed.

        Parameters
        ----------
        root : str
            Path to the directory with the following structure:
                root
                ├── cls1
                ├── cls2
                └── ...

        """
        self.cls = list()
        for cls in os.listdir(root):
            cls_dir = os.path.join(root, cls)
            if os.path.isdir(cls_dir):
                self.cls.append(cls)

    def _get_cls_mapping(self):
        """Get a dictionary that maps each class string with its index"""
        self.cls_mapping = dict(zip(self.cls, range(len(self.cls))))

    def _get_file_paths(self):
        """Get file paths and their labels from a folder with the following
        structure:
            root
            ├── cls1
            │   ├── img1.png # Image can be of any extensions
            │   ├── img2.png
            │   └── ...
            ├── cls2
            │   ├── img1.png # Image can be of any extensions
            │   ├── img2.png
            │   └── ...
            └── ...
        Subclass may overwrite this function if necessary.
        """
        for cls in self.cls:
            if cls not in self.cls:
                self.cls.append(cls)
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                _, ext = os.path.splitext(filename)
                if ext.lower() not in IMG_EXTENSIONS:
                    continue
                # Append
                self.filepaths.append(os.path.join(cls_dir, filename))
                self.labels.append(self.cls_mapping[cls])  # get class index

    def __getitem__(self, idx):
        img = super(BaseDatasetWithLabels, self).__getitem__(idx)
        return img, self.labels[idx]


class BaseDatasetNpy(torch.utils.data.Dataset):
    """
    Base class for any dataset implementation which reads data from numpy
    array of pre-processed images. This base class can be applied to both small
    and large datasets, given that the images are pre-processed (with
    `extract_data.py`)and stored in a *.npy file. Note that this can handle
    dataset consisting of multiple parts.
    """
    def __init__(self, info_path, transform=None):
        """
        Initialize the dataset.

        Parameters
        ----------
        info_path : str
            Path to the `info.json` file generated by `extract_data.py`.
        transform : fn
            A function defined in `data_loaders.pre_processing.py` to be taken
            as the custom image transformation function.

        Attributes
        ----------
        data : torch.Tensor
            All images loaded as torch tensor.
        filepaths : list
            List of paths to all parts.
        info : dict
            A dictionary loaded from `info.json` containing useful information
            about the dataset.
        info_path

        """
        self.info_path = info_path
        self.info = read_json(info_path)
        # Get useful info
        self.dtype = self.info["dtype"]
        self.num_parts = self.info["num_parts"]
        self.num_samples = self.info["num_images"]

        # Get dataset shape and file paths
        parts = self.info["parts"]
        self.filepaths = list()
        self.sample_shape = None

        for part_idx, part_info in parts.items():
            self.filepaths.append(part_info["save_path"])
            s_shape = part_info["shape"][1:]  # sample shape of this part

            if self.sample_shape is None:
                self.sample_shape = s_shape
            # Ensure that all parts have the same sample shape
            else:
                if self.sample_shape != s_shape:
                    raise ValueError(
                        "All parts must have the same sample shape. Got {} and"
                        " {} instead.".format(self.sample_shape, s_shape))

        # Load data directly if there is only one part
        if self.num_parts == 1:
            self.data = torch.from_numpy(np.load(self.filepaths[0]))
        # Otherwise, pre-allocate memory and load parts one by one
        else:
            self.data = np.empty(
                shape=(self.num_samples, *self.sample_shape), dtype=self.dtype)
            end = 0
            for filepath in self.filepaths:
                part_data = np.load(filepath)
                start = end
                end += len(part_data)
                self.data[start:end] = part_data
            self.data = torch.from_numpy(self.data)

        # Transformation function
        self.transform = transform
        if self.transform is None:
            self.transform = lambda x: x  # identity function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

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

        return self.data[idxs]


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
