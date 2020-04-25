"""
Code adapted from https://github.com/abdulfatir/gan-metrics-pytorch
"""

from collections import OrderedDict

import cv2
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet5 implementation for features extraction.

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
    def __init__(self, model_path):
        """Initialize LeNet model.

        Parameters
        ----------
        model_path : str
            Path to the pretrained LeNet5 model.

        """
        super(LeNet5, self).__init__()
        self.model_path = model_path

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

    def forward(self, inp):
        """Get Inception feature maps.

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
            inp = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
        # Resize if needed
        if inp.shape[2] != 32 or inp.shape[3] != 32:
            inp = F.interpolate(
                inp, size=(32, 32), mode='bilinear', align_corners=False)
        output = self.convnet(inp)
        output = output.view(inp.size(0), -1)
        output = self.fc[1](self.fc[0](output))
        return output
