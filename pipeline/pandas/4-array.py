#!/usr/bin/env python3
"""
Module that converts the last 10 rows of High and Close
columns from a DataFrame into a numpy.ndarray.
"""

import pandas as pd
import numpy as np


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns

    Returns:
        numpy.ndarray: Array of the last 10 rows of High and Close
    """
    return df[["High", "Close"]].tail(10).to_numpy()
