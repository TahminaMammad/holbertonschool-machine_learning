#!/usr/bin/env python3
"""
Module to save and load weights of a Keras model
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights

    Args:
        network: the model whose weights should be saved
        filename: path of the file to save the weights to
        save_format: format in which the weights should be saved

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights

    Args:
        network: the model to which the weights should be loaded
        filename: path of the file to load the weights from

    Returns:
        None
    """
    network.load_weights(filename)
    return None
