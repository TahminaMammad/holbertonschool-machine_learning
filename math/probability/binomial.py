#!/usr/bin/env python3
"""
Binomial distribution module
"""


class Binomial:
    """
    Binomial distribution class
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a Binomial distribution
        data: list of data to estimate n and p
        n: number of Bernoulli trials
        p: probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # estimate p first
            p_est = 1 - (variance / mean) if mean != 0 else 0
            # estimate n and round
            n_est = round(mean / p_est) if p_est != 0 else 0
            # recalculate p
            p_est = mean / n_est if n_est != 0 else 0

            self.n = int(n_est)
            self.p = float(p_est)
