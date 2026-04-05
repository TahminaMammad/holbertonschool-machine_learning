#!/usr/bin/env python3
"""Module for convolutional forward propagation"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer

    Parameters:
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b -- numpy.ndarray of shape (1, 1, 1, c_new)
    activation -- activation function
    padding -- "same" or "valid"
    stride -- tuple (sh, sw)

    Returns:
    Output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))

        ph = int(((h_new - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_new - 1) * sw + kw - w_prev) / 2)

    elif padding == "valid":
        ph = pw = 0
        h_new = int((h_prev - kh) / sh) + 1
        w_new = int((w_prev - kw) / sw) + 1

    # Padding
    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output initialization
    Z = np.zeros((m, h_new, w_new, c_new))

    # Convolution
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice_A = A_prev_padded[i,
                                            vert_start:vert_end,
                                            horiz_start:horiz_end,
                                            :]

                    Z[i, h, w, c] = np.sum(
                        slice_A * W[:, :, :, c]
                    ) + b[:, :, :, c]

    # Apply activation
    A = activation(Z)

    return A
