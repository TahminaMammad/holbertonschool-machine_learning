#!/usr/bin/env python3
"""
Module to update weights with Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    """
    m = Y.shape[1]
    # Starting dZ for the last layer (Softmax + Cross-Entropy)
    # dZ = A[L] - Y
    A_final = cache['A' + str(L)]
    dz = A_final - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Calculate gradients for the current layer i
        dW = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            # Backpropagate to the previous layer
            # dA_prev = W.T * dz
            dA_prev = np.matmul(W.T, dz)

            # Apply the dropout mask and scaling from the forward prop
            # This ensures we only backprop through nodes that were "on"
            D_prev = cache['D' + str(i - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob

            # Calculate dz for the next iteration (tanh derivative)
            # g'(Z) = 1 - A^2
            dz = dA_prev * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
