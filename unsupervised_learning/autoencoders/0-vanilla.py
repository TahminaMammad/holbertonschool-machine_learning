#!/usr/bin/env python3
"""Vanilla Autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer
                              in the encoder.
        latent_dims (int): Dimensions of the latent space.

    Returns:
        tuple: encoder, decoder, auto
            encoder: encoder model
            decoder: decoder model
            auto: full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=latent
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
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
