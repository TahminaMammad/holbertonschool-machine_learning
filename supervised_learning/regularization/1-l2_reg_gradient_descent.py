#!/usr/bin/env python3
"""
Module that contains the function l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization

    Parameters:
    Y (numpy.ndarray): one-hot labels (classes, m)
    weights (dict): weights and biases
    cache (dict): activations
    alpha (float): learning rate
    lambtha (float): L2 regularization parameter
    L (int): number of layers
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    # Output layer
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        # Gradient with L2
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            A_prev = cache['A' + str(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))
