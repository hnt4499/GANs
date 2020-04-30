import math

import numpy as np
import torch


"""
For KID score
"""


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


"""
For FID score
All functions are adapted from `tensorflow_gan.eval.classifier_metrics.py`
"""


def _symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Helper function to compute square root of a symmetric matrix.

    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat. Also note that
    this method **only** works for symmetric matrices.

    Parameters
    ----------
    mat : torch.Tensor
        Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.

    Returns
    -------
    torch.Tensor
        Matrix square root of mat.
    """
    # SVD
    u, s, v = torch.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    mask = s >= eps
    s[mask] = torch.sqrt(s[mask])
    return torch.mm(torch.mm(u, torch.diag(s)), v.T)


def _trace_sqrt_product_torch(sigma, sigma_v):
    """Helper function to find the trace of the positive sqrt of product of
    covariance matrices.

    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily)

    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
        => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
        => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
        => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                      = sum(sqrt(eigenvalues(A B B A)))
                                      = sum(eigenvalues(sqrt(A B B A)))
                                      = trace(sqrt(A B B A))
                                      = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
        use the _symmetric_matrix_square_root function to find the roots of
        these matrices.

    Parameters
    ----------
        sigma : torch.Tensor
            A square, symmetric, real, positive semi-definite covariance
            matrix.
        sigma_v : torch.Tensor
            Same as sigma

    Returns
    -------
    float
        The trace of the positive square root of sigma * sigma_v

    """
    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = _symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.mm(sqrt_sigma, torch.mm(sigma_v, sqrt_sigma))

    return _symmetric_matrix_square_root_torch(sqrt_a_sigmav_a).trace()


def calculate_fid(fake_activations, real_activations, device=None):
    """Classifier distance for evaluating a generative model.

    This methods computes the Frechet classifier distance from activations of
    real images and generated images.

    This technique is described in detail in https://arxiv.org/abs/1706.08500.
    Given two Gaussian distribution with means m and m_w and covariance
    matrices C and C_w, this function calculates

                  |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

    which captures how different the distributions of real images and generated
    images (or more accurately, their visual features) are. Note that unlike
    the Inception score, this is a true distance and utilizes information about
    real world images.

    Note that when computed using sample means and sample covariance matrices,
    Frechet distance is biased. It is more biased for small sample sizes. (e.g.
    even if the two distributions are the same, for a small sample size, the
    expected Frechet distance is large). It is important to use the same
    sample size to compute frechet classifier distance when comparing two
    generative models.

    Parameters
    ----------
    fake_activations : torch.Tensor
        2D Tensor containing activations of generated data of shape
        [batch_size, activation_size].
    real_activations : torch.Tensor
        2D Tensor containing activations of real data of shape
        [batch_size, activation_size].

    Returns
    -------
    float
        The Frechet Inception distance. A floating-point scalar of the same
        type as the output of the activations.

    """
    fake_activations = fake_activations.to(device=device)
    real_activations = real_activations.to(device=device)

    assert real_activations.ndim == fake_activations.ndim == 2

    # Compute mean and covariance matrices of activations.
    m = real_activations.mean(axis=0)
    m_w = fake_activations.mean(axis=0)

    num_examples_real = real_activations.shape[0]
    num_examples_fake = fake_activations.shape[0]

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = torch.mm(real_centered.T, real_centered) / (
        num_examples_real - 1)

    fake_centered = fake_activations - m_w
    sigma_w = torch.mm(fake_centered.T, fake_centered) / (
        num_examples_fake - 1)

    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = _trace_sqrt_product_torch(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = (sigma + sigma_w).trace() - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = torch.dist(m, m_w, p=2) ** 2  # square of L2 norm
    fid = trace + mean

    return fid
