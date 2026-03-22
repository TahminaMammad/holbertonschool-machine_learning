#!/usr/bin/env python3
"""
Module that contains the function l2_reg_cost
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
    cost (tf.Tensor): base cost without regularization
    model (tf.keras.Model): model with L2 regularization

    Returns:
    tf.Tensor: total cost for each layer (cost + L2 losses)
    """
    # Get L2 losses from model
    l2_losses = model.losses

    # Add base cost to each L2 loss
    total_costs = [cost + loss for loss in l2_losses]

    return tf.stack(total_costs)
