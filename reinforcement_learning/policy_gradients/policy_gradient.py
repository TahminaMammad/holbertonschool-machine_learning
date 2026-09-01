#!/usr/bin/env python3
"""Policy gradient helper functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities from a state and its weights."""
    scores = np.matmul(matrix, weight)
    probabilities = np.exp(scores)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Compute an action and its Monte Carlo policy gradient."""
    probabilities = policy(state, weight)
    action = np.random.choice(weight.shape[1], p=probabilities)
    actions = np.zeros(weight.shape[1])
    actions[action] = 1
    gradient = np.outer(state, actions - probabilities)
    return action, gradient
