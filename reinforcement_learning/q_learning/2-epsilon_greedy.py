#!/usr/bin/env python3
"""Module for choosing an action using epsilon-greedy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action using the epsilon-greedy strategy."""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
