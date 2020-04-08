from torchvision import datasets, transforms

from base import BaseDataLoader
from . import datasets as custom_datasets
from . import pre_processing


class ImageNetLoader(BaseDataLoader):
    """ImageNet data loader.

    Parameters
    ----------
    data_dir : str
        Root directory of the ImageNet dataset.
    cls : str
        A ImageNet class name, for example "n01531178", which corresponds to
        "goldfinch".
    image_size : int
        Input images will be center-cropped and resized to this `image_size`.
    include_val : bool
        Whether to include images from validation set.
    transform : str
        A string, name of the function in `pre_processing.py` to be taken as
        the custom image transformation function. (default: "DCGAN_transform")
    batch_size : int
        Mini-batch size for training.
    shuffle : bool
        Whether to shuffle the training set.
    num_workers : int
        The number of subprocesses to use for data loading. `0` means that the
        data will be loaded in the main process. (default: `0`)

    """
    def __init__(self, data_dir, cls, image_size=64, include_val=False,
                 transform="DCGAN_transform", batch_size=128, shuffle=True,
                 num_workers=1):
        # Dataset
        self.data_dir = data_dir
        self.dataset = custom_datasets.ImageNetDataset(
            data_dir, cls, image_size=image_size, include_val=include_val,
            transform=transform)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split=0.0, num_workers=num_workers)
