#!/usr/bin/env python3
"""
Deep RNN forward propagation
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Args:
        rnn_cells (list): list of RNNCell instances of length l
        X (ndarray): shape (t, m, i), data to be used
        h_0 (ndarray): shape (l, m, h), initial hidden states

    Returns:
        H (ndarray): all hidden states, shape (t + 1, l, m, h)
        Y (ndarray): all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]

        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)

            H[step + 1, layer] = h_next
            x_t = h_next

        Y[step] = y

    return H, Y
