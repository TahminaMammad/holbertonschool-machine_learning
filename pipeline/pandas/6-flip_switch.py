#!/usr/bin/env python3
"""
Module that sorts a DataFrame in reverse chronological order
and transposes the result.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order
    and transposes it.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column

    Returns:
        pd.DataFrame: Transposed DataFrame sorted in reverse chronological order
    """
    # Sort by Timestamp descending
    df_sorted = df.sort_values("Timestamp", ascending=False)
    # Transpose the DataFrame
    return df_sorted.transpose()
