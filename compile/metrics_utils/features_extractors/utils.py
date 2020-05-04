"""
Helper functions for feature extraction models.
"""


def normalize(x, in_range, out_range):
    """Helper function to normalize a PyTorch tensor from range `in_range` to
    range `out_range`. Useful for normalize inputs of feature extraction
    models.

    Parameters
    ----------
    x : torch.Tensor
        PyTorch tensor to normalize.
    in_range : type
        Range of x before normalized.
    out_range : type
        Range of x after normalized.

    Returns
    -------
    torch.Tensor
        Tensor after normalized.

    """
    in_min, in_max = in_range
    out_min, out_max = out_range
    # Calculate range
    in_r = float(in_max - in_min)
    out_r = float(out_max - out_min)
    # Scale to `out_r`
    x = x / in_r * out_r
    curr_max = in_max / in_r * out_r  # current maximum value of `x`
    # Shift to correct range
    x = x - curr_max + out_max
    return x
