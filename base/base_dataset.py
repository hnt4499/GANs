import os
import math

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
    Note that all datasets in this repository use label `-1` for images without
    labels (e.g., for unsupervised or semi-supervised learning).
    This base class should only be used for small datasets (< 10K). For
    large datasets, see `BaseDatasetNpy`, which reads data from numpy array of
    pre-processed images"""
    def __init__(self, root, transform, drop_labels=None, stratified=True):
        """
        Initialize base dataset with labels.

        Parameters
        ----------
        root : str
            Root directory of the dataset.
        transform : fn
            A function defined in `data_loaders.pre_processing.py` to be taken
            as the custom image transformation function.
        drop_labels : float or None
            If not None, change the labels of a part of the dataset to `-1`,
            meaning no label. The number of such drops is calculated as
            `math.ceil(num_datapoints_with_labels * drop_labels)`.
        stratified : bool
            If True, sampling data points to drop is done in a stratified
            manner. Uniformly otherwise. Note that sampling uniformly is
            faster.

        Attributes
        ----------
        filepaths : list
            List of all training file paths.
        _labels : list
            List of all labels for each image (in the same order) before label
            dropping.
        labels : ndarray
            Array of all labels after label dropping.
        cls : list
            List of all class strings.
        cls_mapping : dict
            A dictionary that maps each class string with its index (label).
        root
        transform
        drop_labels
        stratified

        """
        self._labels = list()
        self.cls = list()
        self.root = root
        # Starting initializing and reading information
        self._get_cls()
        self._get_cls_mapping()
        # Initialize base class
        super(BaseDatasetWithLabels, self).__init__(
            root=root, transform=transform)
        # Drop labels
        self.drop_labels = drop_labels
        self.stratified = stratified
        self._drop_labels()

    def _get_cls(self):
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
        for cls in os.listdir(self.root):
            cls_dir = os.path.join(self.root, cls)
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
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                _, ext = os.path.splitext(filename)
                if ext.lower() not in IMG_EXTENSIONS:
                    continue
                # Append
                self.filepaths.append(os.path.join(cls_dir, filename))
                self._labels.append(self.cls_mapping[cls])  # get class index

    def _drop_labels(self):
        """Drop labels of a part of the dataset"""
        self.labels = np.asarray(self._labels, dtype="int")
        if self.drop_labels is None:
            return
        self.labels = drop_labels_helper(
            self.labels, drop_fraction=self.drop_labels, fill=-1,
            stratified=self.stratified)

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
                del part_data  # for memory efficiency
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


class BaseDatasetWithLabelsNpy(BaseDatasetNpy):
    """
    Base class for any dataset implementation which reads data from numpy
    array of pre-processed images and their labels. This base class can be
    applied to both small and large datasets, given that the images are
    pre-processed (with `extract_data.py`) and stored in a *.npy file. Note
    that this can handle dataset consisting of multiple parts.
    """
    def __init__(self, info_path, transform=None, drop_labels=None,
                 stratified=True):
        """
        Initialize the dataset.

        Parameters
        ----------
        info_path : str
            Path to the `info.json` file generated by `extract_data.py`.
        transform : fn
            A function defined in `data_loaders.pre_processing.py` to be taken
            as the custom image transformation function.
        drop_labels : float or None
            If not None, change the labels of a part of the dataset to `-1`,
            meaning no label. The number of such drops is calculated as
            `math.ceil(num_datapoints_with_labels * drop_labels)`.
        stratified : bool
            If True, sampling data points to drop is done in a stratified
            manner. Uniformly otherwise. Note that sampling uniformly is
            faster.

        Attributes
        ----------
        data : torch.Tensor
            All images loaded as torch tensor.
        filepaths : list
            List of paths to all parts.
        cls : list
            List of all class strings.
        cls_mapping : dict
            A dictionary that maps each class string with its index (label).
        _labels : list
            List of all labels for each image (in the same order) before label
            dropping.
        labels : ndarray
            Array of all labels after label dropping.
        info : dict
            A dictionary loaded from `info.json` containing useful information
            about the dataset.
        info_path

        """
        super(BaseDatasetWithLabelsNpy, self).__init__(
            info_path=info_path, transform=transform)
        self.cls_mapping = self.info["cls_mapping"]
        self.cls = list(self.cls_mapping.keys())
        # Load labels
        self._load_labels()
        # Drop labels
        self.drop_labels = drop_labels
        self.stratified = stratified
        self._drop_labels()

    def _load_labels(self):
        parts = self.info["parts"]
        labels = list()
        for part_idx, part_info in parts.items():
            labels_path = part_info["labels_path"]
            labels.append(np.load(labels_path))
        self._labels = np.concatenate(labels)

    def _drop_labels(self):
        self.labels = np.asarray(self._labels)
        if self.drop_labels is None:
            return
        self.labels = drop_labels_helper(
            self.labels, drop_fraction=self.drop_labels, fill=-1,
            stratified=self.stratified)

    def __getitem__(self, idx):
        img = super(BaseDatasetWithLabelsNpy, self).__getitem__(idx)
        return img, self.labels[idx]


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


"""
Other utilities functions
"""


def drop_labels_helper(labels, drop_fraction=0.0, fill=-1, stratified=True):
    """Change the labels of a part of the dataset to a specified value.

    Parameters
    ----------
    labels : ndarray
        List of labels.
    drop_fraction : float
        Fraction of data (with labels) to be dropped.
    fill : int
        Value to fill.
    stratified : bool
        Whether to sample the labels stratifiedly.

    Returns
    -------
    ndarray
        A new array of labels with some changed to the specified value.

    """
    labels = np.copy(labels)
    cls = np.unique(labels)
    idxs = np.arange(len(labels), dtype="int")
    if stratified:
        for c in cls:
            # Get indices of data belonging to current class
            idxs_with_labels = idxs[labels == c]
            # Get number of datapoints to drop labels from
            num_to_drop = math.ceil(
                len(idxs_with_labels) * drop_fraction)
            # Sample randomly
            to_drop = np.random.choice(
                idxs_with_labels, size=num_to_drop, replace=False)
            # Drop
            labels[to_drop] = -1
    else:
        # Get indices of data with labels
        idxs_with_labels = idxs[labels != -1]
        # Get number of datapoints to drop labels from
        num_to_drop = math.ceil(len(idxs_with_labels) * drop_fraction)
        # Sample randomly
        to_drop = np.random.choice(
            idxs_with_labels, size=num_to_drop, replace=False)
        # Drop
        labels[to_drop] = -1
    return labels
