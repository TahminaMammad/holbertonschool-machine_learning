#!/usr/bin/env python3
"""
Module that slices a DataFrame to extract specific columns
and select every 60th row.
"""

import pandas as pd


def slice(df):
    """
    Extracts the columns High, Low, Close, and Volume_(BTC),
    then selects every 60th row.

    Args:
        df (pd.DataFrame): DataFrame containing the required columns

    Returns:
        pd.DataFrame: Sliced DataFrame with selected columns and rows
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
