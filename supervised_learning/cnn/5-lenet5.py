#!/usr/bin/env python3
"""Module for LeNet-5 architecture using Keras"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 model

    Returns:
    Compiled Keras model
    """

    init = K.initializers.he_normal(seed=0)

    # Conv 1
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Conv 2
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten
    flat = K.layers.Flatten()(pool2)

    # Dense layers
    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flat)

    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(dense1)

    # Output
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(dense2)

    # Model
    model = K.Model(inputs=X, outputs=output)

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
