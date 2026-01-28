#!/usr/bin/env python3
"""
Module 6-flip_switch
Provides a function to sort a DataFrame in reverse chronological order
and transpose it.
"""


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transposed DataFrame after sorting.
    """
    # Sort by index in descending order (reverse chronological)
    df_sorted = df.sort_index(ascending=False)

    # Transpose the DataFrame
    return df_sorted.transpose()
