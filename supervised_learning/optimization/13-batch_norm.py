#!/usr/bin/env python3
"""
Module to perform batch normalization on a matrix
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized
        gamma: numpy.ndarray of shape (1, n) containing scales
        beta: numpy.ndarray of shape (1, n) containing offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # 1. Calculate the mean and variance along the batch (m)
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # 2. Standardize Z
    # Z_centered = (Z - mean) / sqrt(variance + epsilon)
    Z_centered = (Z - mean) / np.sqrt(variance + epsilon)

    # 3. Scale and shift (Reconstruction)
    # Allows the network to learn the optimal distribution
    # Z_tilde = gamma * Z_centered + beta
    Z_tilde = gamma * Z_centered + beta

    return Z_tilde
