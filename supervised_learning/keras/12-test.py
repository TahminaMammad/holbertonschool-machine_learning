#!/usr/bin/env python3
"""
Module to test a Keras model
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: the correct one-hot labels of data
        verbose: boolean determining if output should be printed

    Returns:
        The loss and accuracy of the model respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
