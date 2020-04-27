"""
Code adapted from https://github.com/abdulfatir/gan-metrics-pytorch
"""

from collections import OrderedDict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .base_fe import BaseFeatureExtractor


class LeNet5FE(BaseFeatureExtractor):
    """LeNet5 feature extractor.

    Parameters
    ----------
    model_path : str
        Path to the pretrained model.
    resize_to : int
        If None, do not resize input images.
        If not None, bilinearly resizes input images to width and height
        `resize_to` before feeding input to model.
        As the network without fully connected layers is fully
        convolutional, it should be able to handle inputs of arbitrary
        size, so resizing might not be strictly needed. If needed, the
        images should be resized to (32, 32).

    batch_size : int
        Mini-batch size for feature extractor.
    device : str
        String argument to pass to `torch.device`, for example, "cpu" or
        "cuda".

    Attributes
    ----------
    id_real, id_fake
        An unique ID determining current step of the training process. When
        calling this class instance with a different ID, features will be
        re-computed for the new ID. When calling this class instance with the
        same ID, cached features will be returned.
        This protocol allows sharing features between different metrics by
        using the same ID (for example, current batch index) at a step.
    feats_real : torch.Tensor
        Features extracted from real samples.
    feats_fake : torch.Tensor
        Features extracted from fake samples.
    model : LeNet5
        LeNet5 model.
    batch_size

    """
    def __init__(self, model_path, resize_to=None, batch_size=64,
                 device="cpu"):
        super(LeNet5FE, self).__init__(
            model_initializer=LeNet5, batch_size=batch_size, device=device,
            model_path=model_path, resize_to=resize_to,
        )

    def __eq__(self, other):
        # Check if `other` is an instance of `InceptionV3FE` or its subclass
        if not isinstance(other, LeNet5FE):
            return False
        # Simply compare InceptionV3 parameters
        return super(LeNet5FE, self).__eq__(other)


class LeNet5(nn.Module):
    """
    LeNet5 implementation for feature extraction.

    Input   - 1x32x32
    c1      - 6x28x28 (5x5 kernel)
    tanh
    s2      - 6x14x14 (maxpool 2x2)
    c3      - 16x10x10 (5x5 kernel)
    tanh
    s4      - 16x5x5 (maxpool 2x2)
    c5      - 120x1x1 (5x5 kernel)
    tanh
    flatten - 120
    f6      - 84
    tanh
    f7      - 10
    softmax (output)
    """
    def __init__(self, model_path, resize_to=None):
        """Initialize LeNet model.

        Parameters
        ----------
        model_path : str
            Path to the pretrained LeNet5 model.

        resize_to : int
            If None, do not resize input images.
            If not None, bilinearly resizes input images to width and height
            `resize_to` before feeding input to model.
            As the network without fully connected layers is fully
            convolutional, it should be able to handle inputs of arbitrary
            size, so resizing might not be strictly needed. If needed, the
            images should be resized to (32, 32).

        """
        super(LeNet5, self).__init__()
        self.model_path = model_path
        self.resize_to = resize_to

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('tanh1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('tanh3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('tanh5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('tanh6', nn.Tanh()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

        # Load pretrained model
        self.load_state_dict(torch.load(model_path))
        # Turn off gradients completely
        for param in self.parameters():
            param.requires_grad = False

        # LeNet expects inputs to be grayscale images
        self.transformer = transforms.Compose([
            transforms.ToPILImage(mode="RGB"),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def forward(self, inp):
        """Get LeNet feature maps.

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3x32x32, resized if needed. Values are
            expected to be in range (0.0, 1.0).

        Returns
        -------
        torch.autograd.Variable
            The features extracted from the first fully-connected layer.
        """
        # Convert to grayscale if needed
        if inp.shape[1] == 3:
            inp = self._transform(inp)
        # Resize if needed
        if self.resize_to is not None:
            inp = F.interpolate(
                inp, size=(self.resize_to, self.resize_to), mode='bilinear',
                align_corners=False)
        # Feed forward
        output = self.convnet(inp)
        output = output.view(inp.size(0), -1)
        output = self.fc[1](self.fc[0](output))
        return output

    def _transform(self, inp):
        """Helper function to convert batched RGB images to grayscale"""
        device = inp.device
        # First send images to CPU
        inp = inp.cpu()
        # Transform
        inp = list(map(self.transformer, inp))
        # Stack
        inp = torch.stack(inp)
        # Send back to original device
        inp = inp.to(device=device)

        return inp
