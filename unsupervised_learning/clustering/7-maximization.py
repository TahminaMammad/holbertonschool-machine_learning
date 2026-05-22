#!/usr/bin/env python3
"""Maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set.
        g (numpy.ndarray): shape (k, n) containing posterior probabilities.

    Returns:
        tuple:
            pi (numpy.ndarray): shape (k,) containing updated priors.
            m (numpy.ndarray): shape (k, d) containing updated means.
            S (numpy.ndarray): shape (k, d, d) containing updated covariances.

        Returns (None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2):
        return None, None, None

    if (not isinstance(g, np.ndarray) or g.ndim != 2):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    if np.any(g < 0):
        return None, None, None

    weights = np.sum(g, axis=1)

    pi = weights / n

    m = (g @ X) / weights[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = (weighted_diff.T @ diff) / weights[i]

    return pi, m, S
