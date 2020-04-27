import math

import numpy as np
import torch


def calculate_kid(fake_activations, real_activations, device, batch_size=1024):
    """Adapted from
        https://github.com/google/compare_gan/blob/master/compare_gan/metrics/kid_score.py

    Compute KID score using PyTorch for speed and memory efficiency.

    See `test_kid.ipynb` for performance comparison between different
    implementations (TensorFlow, NumPy, PyTorch CPU and PyTorch GPU).
    Do note that this function gives slightly different (but mathematically
    acceptable) results between CPU and GPU.

    Parameters
    ----------
    fake_activations : torch.Tensor
        Features extracted from fake images.
    real_activations : torch.Tensor
        Features extracted from real images.
    device
        Which device to use. Note that different devices give slightly
        different results.
    batch_size : int
        Features (activations) will be splitted to bins of size `batch_size`.

    Returns
    -------
    float
        Computed KID score (mean and variance) between real and fake images.

    """
    assert fake_activations.ndim == real_activations.ndim == 2

    n_real, dim = real_activations.shape
    n_gen, dim2 = fake_activations.shape
    assert dim2 == dim  # make sure they have the same number of features

    # Split into largest approximately-equally-sized blocks
    n_bins = int(math.ceil(max(n_real, n_gen) / batch_size))
    bins_r = np.full(n_bins, int(math.ceil(n_real / n_bins)))
    bins_g = np.full(n_bins, int(math.ceil(n_gen / n_bins)))
    bins_r[:(n_bins * bins_r[0]) - n_real] -= 1
    bins_g[:(n_bins * bins_r[0]) - n_gen] -= 1
    assert bins_r.min() >= 2
    assert bins_g.min() >= 2
    # Indices of batches
    inds_r = np.r_[0, np.cumsum(bins_r)]
    inds_g = np.r_[0, np.cumsum(bins_g)]

    def get_kid_batch(i):
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = real_activations[r_s:r_e].to(device=device)
        m = r_e - r_s

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = fake_activations[g_s:g_e].to(device=device)
        n = g_e - g_s

        # Could probably do this a bit faster...
        k_rr = (torch.mm(r, r.T) / dim + 1) ** 3
        k_rg = (torch.mm(r, g.T) / dim + 1) ** 3
        k_gg = (torch.mm(g, g.T) / dim + 1) ** 3
        return (
            -2 * k_rg.mean() + (k_rr.sum() - k_rr.trace()) / (m * (m - 1))
            + (k_gg.sum() - k_gg.trace()) / (n * (n - 1))).cpu().numpy()

    ests = map(get_kid_batch, range(n_bins))
    ests = np.asarray(list(ests))
    return ests.mean(), ests.var()
