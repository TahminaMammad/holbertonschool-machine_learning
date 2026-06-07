#!/usr/bin/env python3
"""
Variational Autoencoder implementation
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder (keras.Model): encoder model, outputs latent representation,
                               mean, and log variance
        decoder (keras.Model): decoder model
        auto (keras.Model): full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0., stddev=1.
        )
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # Autoencoder
    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, outputs)

    # Custom loss: reconstruction + KL divergence
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean) -
        keras.backend.exp(z_log_var), axis=1
    )

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto

