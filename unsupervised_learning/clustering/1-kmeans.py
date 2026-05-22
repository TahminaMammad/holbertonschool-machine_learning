#!/usr/bin/env python3
"""Performs K-means clustering."""


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

    return np.random.uniform(low=mins, high=maxs,
                             size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset.
        k (int): number of clusters.
        iterations (int): maximum number of iterations.

    Returns:
        tuple:
            C (numpy.ndarray): centroid means for each cluster.
            clss (numpy.ndarray): index of cluster for each point.
        or (None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None

    if (not isinstance(k, int) or k <= 0):
        return None, None

    if (not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):
            points = X[clss == j]

            if len(points) == 0:
                new_C[j] = np.random.uniform(
                    low=mins,
                    high=maxs,
                    size=(d,)
                )
            else:
                new_C[j] = np.mean(points, axis=0)

        if np.allclose(C, new_C):
            return new_C, clss

        C = new_C

    return C, clss
