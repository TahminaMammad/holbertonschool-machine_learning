#!/usr/bin/env python3
"""
Module that contains the function l2_reg_cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
    cost (float): original cost without regularization
    lambtha (float): regularization parameter
    weights (dict): dictionary of weights and biases
    L (int): number of layers
    m (int): number of data points

    Returns:
    float: total cost with L2 regularization
    """
    l2_sum = 0

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * l2_sum

    return cost + l2_cost
