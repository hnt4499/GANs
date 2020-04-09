import os

import pandas as pd
import torch
from PIL import Image


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
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNetDataset(torch.utils.data.Dataset):
    """ImageNet dataset reader.

    Parameters
    ----------
    root : str
        Root directory of the ImageNet dataset.
    cls : str
        A ImageNet class name, for example "n01531178", which corresponds to
        "goldfinch".
    image_size : int
        Input images will be center-cropped and resized to this `image_size`.
    transform : str
        A function in `pre_processing.py` to be taken as the custom image
        transformation function.
    include_val : bool
        Whether to include images from validation set.

    """
    def __init__(self, root, cls, image_size, transform, include_val=False):
        super(ImageNetDataset).__init__()
        # Get train filepaths
        train_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/train")
        train_dir = os.path.join(train_dir, cls)
        train_names = os.listdir(train_dir)
        train_paths = [os.path.join(train_dir, train_name)
                       for train_name in train_names]
        self.filepaths = train_paths
        # Get validation filenames
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
        # Get transformation function
        self.transform = transform
        # Load images
        self.images = [self.transform(default_loader(filepath))
                       for filepath in self.filepaths]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.images[idx]
