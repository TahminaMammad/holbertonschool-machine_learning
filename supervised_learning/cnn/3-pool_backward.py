#!/usr/bin/env python3
"""Module for pooling backward propagation"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer

    Returns:
    dA_prev
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_prev = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize
    dA_prev = np.zeros_like(A_prev)

    # Loop
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):

                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == "max":
                        slice_A = A_prev[i,
                                         vert_start:vert_end,
                                         horiz_start:horiz_end,
                                         c]

                        mask = (slice_A == np.max(slice_A))

                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += mask * dA[i, h, w, c]

                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        average = da / (kh * kw)

                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += np.ones((kh, kw)) * average

    return dA_prev
