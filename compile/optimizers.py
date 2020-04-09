"""
All functions in this module must return another function that takes only the
model trainable parameters.
"""

from functools import partial

import torch


def Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08,
         weight_decay=0, amsgrad=False):
    return partial(torch.optim.Adam, lr=lr, betas=betas, eps=eps,
                   weight_decay=weight_decay, amsgrad=amsgrad)
