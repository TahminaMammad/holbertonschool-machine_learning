#!/usr/bin/env python3
"""
Expectation-Maximization algorithm for Gaussian Mixture Models
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a Gaussian Mixture Model (GMM)

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points and d is the
        number of dimensions.
    k : int
        The number of clusters.
    iterations : int, optional
        The maximum number of iterations for the algorithm (default 1000).
    tol : float, optional
        Tolerance of the log likelihood for early stopping (default 1e-5).
    verbose : bool, optional
        If True, prints log likelihood information during training.

    Returns
    -------
    pi : numpy.ndarray of shape (k,)
        The priors for each cluster.
    m : numpy.ndarray of shape (k, d)
        The centroid means for each cluster.
    S : numpy.ndarray of shape (k, d, d)
        The covariance matrices for each cluster.
    g : numpy.ndarray of shape (k, n)
        The probabilities for each data point in each cluster.
    l : float
        The log likelihood of the model.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    l_prev = 0.0

    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if abs(l - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}".format(i, l))
            break
        l_prev = l

    # Ensure consistent rounding for reproducibility
    pi = np.round(pi, 8)
    m = np.round(m, 8)
    S = np.round(S, 8)
    g = np.round(g, 8)
    l = round(l, 5)

    return pi, m, S, g, l

