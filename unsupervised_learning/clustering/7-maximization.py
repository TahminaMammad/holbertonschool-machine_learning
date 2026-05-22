#!/usr/bin/env python3
"""Maximization step for Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """Performs the maximization step in EM for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d), dataset.
        g (numpy.ndarray): shape (k, n), posterior probabilities.

    Returns:
        pi (numpy.ndarray): shape (k,), priors.
        m (numpy.ndarray): shape (k, d), means.
        S (numpy.ndarray): shape (k, d, d), covariances.
        (None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(g, np.ndarray) or g.ndim != 2):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # responsibilities per cluster
    weights = np.sum(g, axis=1)

    # avoid division by zero (important fix)
    if np.any(weights == 0):
        return None, None, None

    # priors
    pi = weights / n

    # means
    m = np.dot(g, X) / weights[:, np.newaxis]

    # covariance matrices
    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot((g[i][:, np.newaxis] * diff).T, diff) / weights[i]

    return pi, m, S
