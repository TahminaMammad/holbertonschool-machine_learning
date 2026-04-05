#!/usr/bin/env python3
"""Module for convolutional backward propagation"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer

    Returns:
    dA_prev, dW, db
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Padding
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    # Pad A_prev and dA_prev
    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_prev = np.zeros_like(A_prev)
    dA_prev_padded = np.pad(
        dA_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dW = np.zeros_like(W)
    db = np.zeros((1, 1, 1, c_new))

    # db is simple
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Backprop loop
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

                    # dW
                    dW[:, :, :, c] += slice_A * dZ[i, h, w, c]

                    # dA_prev
                    dA_prev_padded[i,
                                   vert_start:vert_end,
                                   horiz_start:horiz_end,
                                   :] += W[:, :, :, c] * dZ[i, h, w, c]

    # Remove padding
    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
