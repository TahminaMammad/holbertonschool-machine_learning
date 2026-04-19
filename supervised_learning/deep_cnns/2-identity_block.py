#!/usr/bin/env python3
"""
Identity block for ResNet
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in ResNet

    Parameters:
    A_prev -- output from previous layer
    filters -- tuple/list of (F11, F3, F12)

    Returns:
    Activated output of the identity block
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # Save input for shortcut connection
    shortcut = A_prev

    # First component: 1x1 conv
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 conv
    X = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 conv
    X = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
