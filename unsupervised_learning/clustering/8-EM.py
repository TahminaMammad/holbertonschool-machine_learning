i#!/usr/bin/env python3
"""Expectation Maximization algorithm for a Gaussian Mixture Model."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform the expectation maximization algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.
        iterations: positive integer, maximum number of iterations.
        tol: non-negative float, tolerance of the log likelihood for early
             stopping. Stops if the difference is less than or equal to tol.
        verbose: boolean; if True, prints log likelihood every 10 iterations
                 and after the last iteration.

    Returns:
        pi: numpy.ndarray of shape (k,) with priors for each cluster.
        m: numpy.ndarray of shape (k, d) with centroid means for each cluster.
        S: numpy.ndarray of shape (k, d, d) with covariance matrices.
        g: numpy.ndarray of shape (k, n) with probabilities per data point.
        l: log likelihood of the model.
        Returns None, None, None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    l_prev = 0

    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        if i > 0 and abs(l - l_prev) <= tol:
            break

        l_prev = l
        pi, m, S = maximization(X, g)

    g, l = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))

    return pi, m, S, g, l
