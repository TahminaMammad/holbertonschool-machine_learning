#!/usr/bin/env python3
"""Module for calculating positional encodings."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate the positional encoding for a transformer.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth.

    Returns:
        A numpy array of shape (max_seq_len, dm) containing
        the positional encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angles = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / np.float64(dm)
    )

    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding
