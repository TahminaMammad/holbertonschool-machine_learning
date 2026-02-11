#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """
    Normal distribution class
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize a Normal distribution
        data: list of data to estimate mean and stddev
        mean: given mean if data is None
        stddev: given stddev if data is None
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float(
                (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5
            )

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        return coefficient * (Normal.e ** exponent)
