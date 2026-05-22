#!/usr/bin/env python3
"""Initialize cluster centroids for K-means clustering."""


import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset.
        k (int): number of clusters.

    Returns:
        numpy.ndarray: shape (k, d) containing initialized centroids,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(low=mins, high=maxs, size=(k, X.shape[1]))
