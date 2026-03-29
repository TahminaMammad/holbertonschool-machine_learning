#!/usr/bin/env python3
"""Module for performing convolution with custom padding on grayscale images"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    Args:
        images (numpy.ndarray): shape (m, h, w)
            containing multiple grayscale images
        kernel (numpy.ndarray): shape (kh, kw)
            containing the kernel for the convolution
        padding (tuple): (ph, pw)
            ph is padding for height
            pw is padding for width

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
