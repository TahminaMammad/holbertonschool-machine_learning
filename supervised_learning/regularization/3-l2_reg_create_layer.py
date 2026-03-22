#!/usr/bin/env python3
"""
Module that contains the function l2_reg_create_layer
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization

    Parameters:
    prev (tf.Tensor): output of the previous layer
    n (int): number of nodes in the layer
    activation: activation function to use
    lambtha (float): L2 regularization parameter

    Returns:
    tf.Tensor: output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )

    return layer(prev)
