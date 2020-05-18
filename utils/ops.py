"""Utilities functions for PyTorch operations."""

import torch.nn as nn


class View(nn.Module):
    """A module for dynamic tensor reshaping."""
    def __init__(self, *shape):
        """Initialize View object.

        Parameters
        ----------
        shape : list
            List of shapes for each dimension. Identical behavior to
            `tensor.view()`

        """
        super(View, self).__init__()
        self.shape = shape

    def __repr__(self):
        return 'View{}'.format(tuple(self.shape))

    def forward(self, inp):
        out = inp.view(*self.shape)
        return out
