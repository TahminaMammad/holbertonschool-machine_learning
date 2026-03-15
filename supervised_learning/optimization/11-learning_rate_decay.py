#!/usr/bin/env python3
"""
Module to update learning rate using inverse time decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Args:
        alpha: the original learning rate
        decay_rate: weight used to determine the rate of decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # Calculate the number of times the decay has been applied
    # Using integer division // ensures the "stepwise" behavior
    decay_factor = 1 + decay_rate * (global_step // decay_step)

    # Apply inverse time decay formula
    alpha_decayed = alpha / decay_factor

    return alpha_decayed
