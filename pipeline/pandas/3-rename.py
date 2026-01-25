#!/usr/bin/env python3
"""
Module that renames the Timestamp column to Datetime,
converts values to datetime, and displays only Datetime and Close.
"""

import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts values to datetime,
    and returns only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): DataFrame containing a column named 'Timestamp'

    Returns:
        pd.DataFrame: Modified DataFrame with 'Datetime' and 'Close' columns
    """
    # Rename column
    df = df.rename(columns={"Timestamp": "Datetime"})
    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    # Return only Datetime and Close
    return df[["Datetime", "Close"]]
