#!/usr/bin/env python3
"""Module for performing pooling on images"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        numpy.ndarray: pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            region = images[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
