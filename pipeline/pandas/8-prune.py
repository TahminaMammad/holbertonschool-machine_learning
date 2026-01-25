#!/usr/bin/env python3
"""
Module that removes entries where Close has NaN values.
"""

import pandas as pd


def prune(df):
    """
    Removes any entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Close' column

    Returns:
        pd.DataFrame: Modified DataFrame with NaN values in Close removed
    """
    return df.dropna(subset=["Close"])
