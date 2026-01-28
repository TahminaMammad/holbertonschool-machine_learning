#!/usr/bin/env python3
"""
Module that rearranges and concatenates DataFrames
using a hierarchical index.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two DataFrames with a hierarchical index.

    - Indexes both DataFrames on the Timestamp column.
    - Selects rows between timestamps 1417411980 and 1417417980.
    - Concatenates bitstamp and coinbase data with keys.
    - Rearranges the MultiIndex so Timestamp is the first level.
    - Sorts the data chronologically.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame with hierarchical index.
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select the required timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys
    df = pd.concat(
        [df2, df1],
        keys=['bitstamp', 'coinbase']
    )

    # Rearrange MultiIndex to have Timestamp first
    df = df.swaplevel(0, 1)

    # Sort by Timestamp
    df = df.sort_index()

    return df
