#!/usr/bin/env python3
"""Determines the optimum number of clusters by variance."""


import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        kmin (int): minimum number of clusters to check
        kmax (int): maximum number of clusters to check
        iterations (int): maximum number of iterations for K-means

    Returns:
        results, d_vars

        results is a list containing the outputs of K-means
        for each cluster size

        d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size

        Returns (None, None) on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None

    if (not isinstance(kmin, int) or
            kmin <= 0):
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if (not isinstance(kmax, int) or
            kmax <= 0 or
            kmax <= kmin):
        return None, None

    if (not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    base_var = variances[0]

    d_vars = [base_var - var for var in variances]

    return results, d_vars
