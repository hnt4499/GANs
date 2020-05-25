"""Utilities functions for PyTorch operations."""

import torch
import torch.nn as nn

"""
PyTorch layers
"""
BatchNorm2d = nn.BatchNorm2d


"""
Custom layers.
"""


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
        return "View{}".format(tuple(self.shape))

    def forward(self, inp, **kwargs):
        shape = list()
        for s in self.shape:
            if isinstance(s, int):
                shape.append(s)
            else:
                s = int(s)
                shape.append(inp.shape[s])
        out = inp.view(*shape)
        return out


class MinibatchDiscriminationV1(nn.Module):
    """
    Minibatch discrimination layer. Note that this implementation is slightly
    different from the Lasagne implementation (although it works exactly as
    described in the paper).

    Adapted from:
        https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
    Reference:
        Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
        Chen, X. (2016). Improved Techniques for Training GANs.

    """
    def __init__(self, in_features, out_features, kernel_dims=5, mean=True):
        """Initialize minibatch discrimination layer.

        Parameters
        ----------
        in_features : int
            Input features length. `A` in the paper.
        out_features : int
            Output features length (and also number of kernels). `B` in the
            paper.
        kernel_dims : int
            Kernel dimension. `C` in the paper.
        mean : bool
            Whether to average features along the inputs. (default: True)

        """
        super(MinibatchDiscriminationV1, self).__init__()
        # Cache data
        self.in_features = in_features
        self.out_features_ = out_features
        self.out_features = in_features + out_features  # bc of concatenation
        self.kernel_dims = kernel_dims
        self.mean = mean
        # Initialize parameters
        self.init()

    def __repr__(self):
        return ("MinibatchDiscriminationV1({}, {}, kernel_dims={}, "
                "mean={})".format(self.in_features, self.out_features_,
                                  self.kernel_dims, self.mean))

    def init(self):
        """Initialize parameters"""
        self.theta = nn.Parameter(
            torch.Tensor(self.in_features, self.out_features_,
                         self.kernel_dims))
        nn.init.normal_(self.theta, mean=0, std=1)

    def forward(self, inp, **kwargs):
        # x.shape = (N, A); self.theta.shape = (A, B, C)
        activations = inp.mm(
            self.theta.view(self.in_features, -1))  # (N, B * C)
        activations = activations.view(
            -1, self.out_features_, self.kernel_dims)  # (N, B, C)

        M = activations.unsqueeze(0)  # (1, N, B, C)
        M_T = M.permute(1, 0, 2, 3)  # (N, 1, B, C)
        abs_dif = (M - M_T).abs().sum(3)  # (N, N, B), L2 norm
        exp_abs_dif = (-abs_dif).exp()  # (N. N, B), negative exponential
        mb_feats = (exp_abs_dif.sum(0) - 1)  # (N, B), subtract self distance
        if self.mean:
            mb_feats /= inp.size(0) - 1

        out = torch.cat([inp, mb_feats], 1)
        return out
