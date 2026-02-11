#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution

        Args:
            data (list): data to estimate lambtha
            lambtha (float): expected number of occurrences

        Raises:
            TypeError: if data is not a list
            ValueError: if data contains less than two values
            ValueError: if lambtha is not positive
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            float: PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return ((self.lambtha ** k) *
                (e ** (-self.lambtha)) /
                factorial)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            float: CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf_sum = 0
        for i in range(0, k + 1):
            cdf_sum += self.pmf(i)

        return cdf_sum
