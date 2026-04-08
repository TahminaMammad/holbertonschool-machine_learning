#!/usr/bin/env python3
"""
Module that contains a function to adjust
the hue of an image using TensorFlow.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: a 3D tf.Tensor containing the image to change
        delta: the amount the hue should change

    Returns:
        The hue-adjusted image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
