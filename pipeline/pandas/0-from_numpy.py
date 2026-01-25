#!/usr/bin/env python3
"""
Task 0
Provides a function to create a pandas DataFrame from a NumPy array
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a NumPy ndarray.

    Args:
        array (np.ndarray): NumPy array to convert

    Returns:
        pd.DataFrame: DataFrame with columns labeled A, B, C, ...
    """
    num_cols = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_cols)]
    return pd.DataFrame(array, columns=columns)
