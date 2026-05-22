#!/usr/bin/env python3
"""Maximization step for the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster.

    Returns:
        pi: numpy.ndarray of shape (k,) with updated priors for each cluster.
        m: numpy.ndarray of shape (k, d) with updated centroid means.
        S: numpy.ndarray of shape (k, d, d) with updated covariance matrices.
        Returns None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    if not np.isclose(g.sum(axis=0), 1).all():
        return None, None, None

    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        nk = g[i].sum()
        pi[i] = nk / n
        m[i] = (g[i] @ X) / nk
        diff = X - m[i]
        S[i] = (g[i] * diff.T) @ diff / nk

    return pi, m, S
