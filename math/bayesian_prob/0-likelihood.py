#!/usr/bin/env python3
"""Module for calculating likelihood of a binomial distribution."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x and n
    for various hypothetical probabilities in P.

    Parameters
    ----------
    x : int
        Number of patients that develop severe side effects.
    n : int
        Total number of patients observed.
    P : numpy.ndarray
        1D array of hypothetical probabilities for developing
        severe side effects.

    Returns
    -------
    numpy.ndarray
        1D array containing the likelihood of obtaining the data
        for each probability in P.
    """
    # Input validation
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial likelihood calculation: C(n, x) * P^x * (1-P)^(n-x)
    comb = np.math.comb(n, x)
    likelihoods = comb * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
