#!/usr/bin/env python3
"""
Module that sorts a DataFrame by the High price in descending order.
"""

import pandas as pd


def high(df):
    """
    Sorts the DataFrame by the High column in descending order.

    Args:
        df (pd.DataFrame): DataFrame containing a 'High' column

    Returns:
        pd.DataFrame: Sorted DataFrame
    """
    return df.sort_values("High", ascending=False)
