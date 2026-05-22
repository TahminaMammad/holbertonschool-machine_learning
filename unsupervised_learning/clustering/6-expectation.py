#!/usr/bin/env python3
"""Calculates the expectation step in the EM algorithm."""


import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        pi (numpy.ndarray): shape (k,) containing priors
        m (numpy.ndarray): shape (k, d) containing centroid means
        S (numpy.ndarray): shape (k, d, d) containing covariance matrices

    Returns:
        g, log_likelihood

        g is a numpy.ndarray of shape (k, n)
        containing posterior probabilities

        log_likelihood is the total log likelihood

        Returns (None, None) on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None

    if (not isinstance(pi, np.ndarray) or
            len(pi.shape) != 1):
        return None, None

    if (not isinstance(m, np.ndarray) or
            len(m.shape) != 2):
        return None, None

    if (not isinstance(S, np.ndarray) or
            len(S.shape) != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (m.shape != (k, d) or
            S.shape != (k, d, d)):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    if np.any(pi < 0):
        return None, None

    probs = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        probs[i] = pi[i] * P

    total = np.sum(probs, axis=0)

    g = probs / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
