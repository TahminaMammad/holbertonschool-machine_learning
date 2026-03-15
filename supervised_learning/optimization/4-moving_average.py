#!/usr/bin/env python3
"""
Module to calculate exponentially weighted moving averages
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average (momentum)

    Returns:
        A list containing the moving averages of data
    """
    moving_averages = []
    v = 0  # Initialize moving average value

    for i, x in enumerate(data):
        # Calculate the exponentially weighted average
        # v_t = beta * v_{t-1} + (1 - beta) * theta_t
        v = (beta * v) + ((1 - beta) * x)

        # Apply bias correction
        # v_corrected = v_t / (1 - beta^t)
        v_corrected = v / (1 - (beta ** (i + 1)))

        moving_averages.append(v_corrected)

    return moving_averages
