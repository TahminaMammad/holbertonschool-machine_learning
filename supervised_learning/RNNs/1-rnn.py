#!/usr/bin/env python3
"""
Simple RNN forward propagation
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    Args:
        rnn_cell (RNNCell): instance of RNNCell used for forward propagation
        X (ndarray): shape (t, m, i), data to be used
        h_0 (ndarray): shape (m, h), initial hidden state
    Returns:
        H (ndarray): all hidden states, shape (t + 1, m, h)
        Y (ndarray): all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.Wy.shape[1]

    # Initialize arrays for hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Iterate through time steps
    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H, Y

