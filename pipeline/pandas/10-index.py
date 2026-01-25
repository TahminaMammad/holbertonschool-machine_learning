#!/usr/bin/env python3
"""
Module that sets the Timestamp column as the index of a DataFrame.
"""

import pandas as pd


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column

    Returns:
        pd.DataFrame: Modified DataFrame with Timestamp as index
    """
    return df.set_index("Timestamp")
