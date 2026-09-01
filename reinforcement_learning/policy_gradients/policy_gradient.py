#!/usr/bin/env python3
"""Policy gradient helper functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities from a state and its weights."""
    scores = np.matmul(matrix, weight)
    probabilities = np.exp(scores)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)
