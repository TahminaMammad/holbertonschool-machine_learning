#!/usr/bin/env python3
"""
Module to determine if gradient descent should stop early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        A boolean of whether the network should be stopped early,
        followed by the updated count
    """
    # Check if the improvement is greater than the threshold
    # Improvement = opt_cost - current_cost
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    # If count reaches or exceeds patience, stop early
    if count >= patience:
        return True, count

    return False, count
