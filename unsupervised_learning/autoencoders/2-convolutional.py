#!/usr/bin/env python3
"""Convolutional Autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the input.
        filters (list): Number of filters for each encoder layer.
        latent_dims (tuple): Dimensions of latent representation.

    Returns:
        tuple: encoder, decoder, auto
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=x
    )

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    reversed_filters = filters[::-1]

    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.UpSampling2D(
            size=(2, 2)
        )(x)

    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        size=(2, 2)
    )(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Autoencoder
    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))

    auto = keras.Model(
        inputs=auto_input,
        outputs=auto_output
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
