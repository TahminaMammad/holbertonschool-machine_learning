#!/usr/bin/env python3
"""Calculates total intra-cluster variance for a dataset."""


import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        C (numpy.ndarray): shape (k, d) containing cluster centroids

    Returns:
        float: total intra-cluster variance
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None

    if (not isinstance(C, np.ndarray) or
            len(C.shape) != 2):
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    min_distances = np.min(distances, axis=1)

    return np.sum(min_distances)
