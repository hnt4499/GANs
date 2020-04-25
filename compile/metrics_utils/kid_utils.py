"""
Code adapted from https://github.com/abdulfatir/gan-metrics-pytorch
"""

import math
from collections import OrderedDict

import numpy as np
import torch

from .features_extractors import InceptionV3, LeNet5


def calculate_kid_score(real_images, fake_images, model, device,
                        fe_batch_size=64, kid_batch_size=1024):
    """Calculates the KID score given two sets of images.

    Parameters
    ----------
    real_images : numpy.ndarray
        Input array of shape Bx3xHxW in random order. Values are expected to
        be in range (0.0, 1.0).
    fake_images : numpy.ndarray
        Input array of shape Bx3xHxW in random order. Values are expected to
        be in range (0.0, 1.0).
    model
        Model used for features extraction.
    device
    fe_batch_size : int
        Batch size for features extractors.
    kid_batch_size : int
        Batch size for KID calculation.

    Returns
    -------
    float, float
        Mean and variance of KID score over mini batches.

    """

    # Extract features
    feats_real = get_features(
        real_images, model, device, batch_size=fe_batch_size)
    feats_fake = get_features(
        fake_images, model, device, batch_size=fe_batch_size)
    # Compute KID score
    kid_score = calculate_kid(
        feats_fake, feats_real, batch_size=kid_batch_size)
    return kid_score


def get_features(images, model, device, batch_size=64):
    """Helper function to extract the features (activations) for all images.

    Parameters
    ----------
    images : numpy.ndarray
        Input array of shape Bx3xHxW. Values are expected to be in range
        (0.0, 1.0).
    model
        Model used for features extraction.
    device
    batch_size : int
        Evaluation batch size.

    Returns
    -------
    numpy.ndarray
        Extracted features of shape (num_samples, num_features).

    """

    num_batches = math.ceil(len(images) / batch_size)
    feats = list()

    model.eval()
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size

        images_batch = torch.from_numpy(images[start:end]).to(device=device)
        feat = model(images_batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose too shallow layer.
        shape = list(feat.shape)
        if any(s != 1 for s in shape[2:]):
            feat = adaptive_avg_pool2d(feat, output_size=(1, 1))
        # Flatten feature(s)
        feat = feat.cpu().data.numpy().reshape(shape[0], -1)
        feats.append(feat)

    return np.vstack(feats)


def calculate_kid(fake_activations, real_activations, batch_size=1024):
    """Adapted from
        https://github.com/google/compare_gan/blob/master/compare_gan/metrics/kid_score.py

    Parameters
    ----------
    fake_activations : np.ndarray
        Features extracted from fake images.
    real_activations : type
        Features extracted from real images.
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
    assert dim2 == dim

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
        r = real_activations[r_s:r_e]
        m = r_e - r_s

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = fake_activations[g_s:g_e]
        n = g_e - g_s

        # Could probably do this a bit faster...
        k_rr = (np.dot(r, r.T) / dim + 1) ** 3
        k_rg = (np.dot(r, g.T) / dim + 1) ** 3
        k_gg = (np.dot(g, g.T) / dim + 1) ** 3
        return (
            -2 * k_rg.mean() + (k_rr.sum() - k_rr.trace()) / (m * (m - 1))
            + (k_gg.sum() - k_gg.trace()) / (n * (n - 1)))

    ests = map(get_kid_batch, range(n_bins))
    ests = np.asarray(list(ests))
    return ests.mean(), ests.var()
