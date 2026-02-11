#!/usr/bin/env python3
"""
Module that calculates the intersection of obtaining data with
various hypothetical probabilities using Bayes' rule.
"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining x and n with each
    probability in P.

    Parameters
    ----------
    x : int
        Number of patients that develop severe side effects.
    n : int
        Total number of patients observed.
    P : numpy.ndarray
        1D array of hypothetical probabilities of developing
        severe side effects.
    Pr : numpy.ndarray
        1D array of prior beliefs of P.

    Returns
    -------
    numpy.ndarray
        1D array containing the intersection of obtaining x and n
        with each probability in P, respectively.

    Raises
    ------
    ValueError
        If n is not a positive integer.
        If x is not an integer >= 0.
        If x > n.
        If any value in P or Pr is not in [0, 1].
        If Pr does not sum to 1.
    TypeError
        If P is not a 1D numpy.ndarray.
        If Pr is not a numpy.ndarray with the same shape as P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Binomial PMF
    comb = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))

    return likelihood * Pr
