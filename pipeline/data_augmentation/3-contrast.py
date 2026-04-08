#!/usr/bin/env python3
"""
Module that contains a function to randomly adjust
the contrast of an image using TensorFlow.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: a 3D tf.Tensor representing the input image
        lower: a float representing the lower bound of the
               contrast factor range
        upper: a float representing the upper bound of the
               contrast factor range

    Returns:
        The contrast-adjusted image as a tf.Tensor
    """
    return tf.image.random_contrast(image, lower, upper)
