#!/usr/bin/env python3
"""Module for calculating the intersection of likelihood and prior."""

import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data x and n
    with the various hypothetical probabilities P and prior Pr.

    Parameters
    ----------
    x : int
        Number of patients that develop severe side effects.
    n : int
        Total number of patients observed.
    P : numpy.ndarray
        1D array of hypothetical probabilities for developing
        severe side effects.
    Pr : numpy.ndarray
        1D array containing the prior beliefs of P.

    Returns
    -------
    numpy.ndarray
        1D array containing the intersection of obtaining the
        data with each probability in P.
    """
    # Input validation
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute likelihood using the previous function
    like = likelihood(x, n, P)

    # Intersection = likelihood * prior
    inter = like * Pr

    return inter
