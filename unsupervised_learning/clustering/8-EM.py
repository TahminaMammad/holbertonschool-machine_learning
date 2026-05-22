#!/usr/bin/env python3
"""Expectation-Maximization algorithm for a Gaussian Mixture Model."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d), dataset.
        k (int): number of clusters.
        iterations (int): max number of iterations.
        tol (float): tolerance for early stopping.
        verbose (bool): print log likelihood.

    Returns:
        pi, m, S, g, l or None on failure.
    """

    if (not isinstance(X, np.ndarray) or X.ndim != 2):
        return None, None, None, None, None

    try:
        pi, m, S = initialize(X, k)
    except Exception:
        return None, None, None, None, None

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = None

    for i in range(iterations + 1):

        g, l = expectation(X, pi, m, S)

        if g is None or l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if prev_l is not None and abs(l - prev_l) <= tol:
            break

        prev_l = l

        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

    return pi, m, S, g, l
