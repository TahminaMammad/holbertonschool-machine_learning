#!/usr/bin/env python3
"""
Module that contains the function dropout_forward_prop
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Parameters:
    X (numpy.ndarray): input data (nx, m)
    weights (dict): weights and biases
    L (int): number of layers
    keep_prob (float): probability of keeping a neuron active

    Returns:
    dict: cache containing activations and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Output layer (softmax)
            exp_Z = np.exp(Z)
            cache['A' + str(i)] = exp_Z / np.sum(exp_Z, axis=0)
        else:
            # Hidden layer (tanh)
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob

            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
