#!/usr/bin/env python3
"""
Module to build the ResNet-50 architecture
"""
from tensorflow import keras as K


identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
        The keras model
    """
    initializer = K.initializers.HeNormal(seed=0)
    img_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Initial Conv and Pool
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=initializer)(img_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2: 3 blocks (1 projection, 2 identity)
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3: 4 blocks (1 projection, 3 identity)
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4: 6 blocks (1 projection, 5 identity)
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5: 3 blocks (1 projection, 2 identity)
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Final Average Pooling
    X = K.layers.AveragePooling2D((7, 7), padding='same')(X)

    # Output Layer
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    model = K.models.Model(inputs=img_input, outputs=X)

    return model
