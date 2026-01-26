#!/usr/bin/env python3
"""
Module that concatenates two DataFrames (bitstamp and coinbase)
with proper indexing and labeling.
"""


def concat(df1, df2):
    """
    Indexes both DataFrames on their Timestamp columns,
    selects rows from df2 up to and including timestamp 1417411920,
    concatenates them on top of df1, and labels them with keys.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame
        df2 (pd.DataFrame): Bitstamp DataFrame

    Returns:
        pd.DataFrame: Concatenated DataFrame with keys
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to and including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
