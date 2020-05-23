"""Utilities functions for PyTorch operations."""

import torch.nn as nn


class View(nn.Module):
    """A module for dynamic tensor reshaping."""
    def __init__(self, *shape):
        """Initialize View object.

        Parameters
        ----------
        shape : list of integers and integer strings
            List of shapes for each dimension. `int` will be parsed as the
            shape for that dimension, while `str` (integer string, e.g., "0")
            will be used as the index of the dimension of the tensor being
            passed, hence that dimension's shape will be reserved.

        """
        super(View, self).__init__()
        # Check validity
        for s in shape:
            if isinstance(s, int):
                continue
            elif isinstance(s, str):
                if not s.isnumeric():
                    raise ValueError("Invalid integer string: {}".format(s))
            else:
                raise ValueError("Invalid data type. Expected one of "
                                 "[int, str], got {} instead.".format(type(s)))
        self.shape = shape

    def __repr__(self):
        return 'View{}'.format(tuple(self.shape))

    def forward(self, inp):
        shape = list()
        for s in self.shape:
            if isinstance(s, int):
                shape.append(s)
            else:
                s = int(s)
                shape.append(inp.shape[s])
        out = inp.view(*shape)
        return out
