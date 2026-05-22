#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM model selection."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best k using BIC for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d), dataset.
        kmin (int): minimum number of clusters.
        kmax (int): maximum number of clusters.
        iterations (int): EM iterations.
        tol (float): EM tolerance.
        verbose (bool): print EM logs.

    Returns:
        best_k, best_result, l, b
        or (None, None, None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin < 1 or kmin > kmax:
        return None, None, None, None

    ks = list(range(kmin, kmax + 1))

    likelihoods = []
    bics = []

    best_bic = None
    best_k = None
    best_result = None

    for k in ks:
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        likelihoods.append(l)

        p = (k * d) + (k * d * d) + (k - 1)
        bic = p * np.log(n) - 2 * l
        bics.append(bic)

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(likelihoods), np.array(bics)
