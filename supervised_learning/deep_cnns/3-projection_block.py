#!/usr/bin/env python3
"""
Projection block for ResNet
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in ResNet

    Parameters:
    A_prev -- output from previous layer
    filters -- tuple/list of (F11, F3, F12)
    s -- stride of the first convolution

    Returns:
    Activated output of the projection block
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # Main path
    X = K.layers.Conv2D(F11, (1, 1),
                        strides=(s, s),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    X = K.layers.Conv2D(F3, (3, 3),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    X = K.layers.Conv2D(F12, (1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path (AFTER main path to match naming order)
    shortcut = K.layers.Conv2D(F12, (1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add and activate
    X = K.layers.Add()([X, shortcut])
    X = K.layers.ReLU()(X)

    return X
