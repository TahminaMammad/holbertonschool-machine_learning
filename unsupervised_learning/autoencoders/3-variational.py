#!/usr/bin/env python3
"""
Module to create a Variational Autoencoder (VAE) using Keras.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes
                              for each hidden layer in the encoder.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # ENCODER
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """
        Samples from the latent distribution using reparameterization.
        """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )

    encoder = keras.Model(
        encoder_inputs, [z, z_mean, z_log_var], name='encoder'
    )

    # DECODER
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # AUTOENCODER
    auto_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(encoder_inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
