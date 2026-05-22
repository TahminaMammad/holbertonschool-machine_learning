#!/usr/bin/env python3
"""Performs K-means clustering."""


import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        numpy.ndarray: initialized centroids of shape (k, d)
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(mins, maxs, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters
        iterations (int): maximum number of iterations

    Returns:
        C, clss
        C is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster

        clss is a numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to

        Returns (None, None) on failure
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

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for i in range(iterations):
        distances = np.sqrt(
            np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
        )

        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):
            points = X[clss == j]

            if len(points) == 0:
                new_C[j] = np.random.uniform(
                    low,
                    high,
                    (d,)
                )
            else:
                new_C[j] = np.mean(points, axis=0)

        if np.array_equal(C, new_C):
            return C, clss

        C = np.copy(new_C)

    distances = np.sqrt(
        np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    )

    clss = np.argmin(distances, axis=1)

    return C, clss
