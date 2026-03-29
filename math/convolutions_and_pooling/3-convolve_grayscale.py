#!/usr/bin/env python3
"""Module for performing strided convolution on grayscale images"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = (h - 1) * sh + kh - h
        pw = (w - 1) * sw + kw - w

        ph = ph // 2 + (ph % 2)
        pw = pw // 2 + (pw % 2)

    elif padding == 'valid':
        ph, pw = 0, 0

    else:
        ph, pw = padding

    # Apply padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
