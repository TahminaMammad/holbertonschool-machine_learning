#!/usr/bin/env python3
"""Variational Autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Input dimensions.
        hidden_layers (list): Hidden layer sizes for encoder.
        latent_dims (int): Latent space dimensions.

    Returns:
        tuple: encoder, decoder, auto
    """
    # =====================
    # Encoder
    # =====================
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    mu = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sample(args):
        """Reparameterization trick."""
        mu, log_var = args

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu)
        )

        return (
            mu +
            keras.backend.exp(log_var / 2) * epsilon
        )

    z = keras.layers.Lambda(sample)(
        [mu, log_var]
    )

    encoder = keras.Model(
        encoder_input,
        [z, mu, log_var]
    )

    # =====================
    # Decoder
    # =====================
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_input,
        decoder_output
    )

    # =====================
    # Autoencoder
    # =====================
    auto_output = decoder(z)

    auto = keras.Model(
        encoder_input,
        auto_output
    )

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var
        - keras.backend.square(mu)
        - keras.backend.exp(log_var),
        axis=-1
    )

    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
