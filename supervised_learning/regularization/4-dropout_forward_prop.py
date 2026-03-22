#!/usr/bin/env python3
"""
Module to conduct forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
        weights: dictionary of the weights and biases of the network
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the
        dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear Step: Z = W * A_prev + b
        Z = np.matmul(W, A_prev) + b

        if i < L:
            # Activation Step: tanh for hidden layers
            A = np.tanh(Z)

            # Dropout Step
            # Create a mask of zeros and ones based on keep_prob
            # The mask has the same shape as the activation output A
            mask = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob)
            mask = mask.astype(int)

            # Apply mask and scale (Inverted Dropout)
            A = (A * mask) / keep_prob

            cache['D' + str(i)] = mask
            cache['A' + str(i)] = A
        else:
            # Activation Step: softmax for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(i)] = A

    return cache
