#!/usr/bin/env python3
"""
Module that computes descriptive statistics for all columns
except the Timestamp column.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing cryptocurrency data

    Returns:
        pd.DataFrame: DataFrame of descriptive statistics
    """
    # Drop Timestamp column if it exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Compute descriptive statistics
    return df.describe()
