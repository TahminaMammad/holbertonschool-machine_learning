#!/usr/bin/env python3
"""
Exponential distribution module
"""


class Exponential:
    """
    Exponential distribution class
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize exponential distribution
        data: list of data to estimate lambda
        lambtha: rate parameter if data is not provided
        """
        if data is None:
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list or len(data) < 2:
                raise ValueError(
                    "data must be a list with at least two values"
                )
            self.lambtha = 1 / (sum(data) / len(data))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period x
        Returns 0 if x is out of range
        """
        if x < 0:
            return 0
        return 1 - Exponential.e ** (-self.lambtha * x)
