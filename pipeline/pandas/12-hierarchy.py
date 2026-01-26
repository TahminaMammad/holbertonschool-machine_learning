#!/usr/bin/env python3
"""
Module that rearranges the MultiIndex so Timestamp is the first level,
concatenates bitstamp and coinbase tables within a given range,
and ensures chronological order.
"""


index = __import__("10-index").index


def hierarchy(df1, df2):
    """
    Rearranges the MultiIndex so Timestamp is the first level,
    concatenates bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980 inclusive, and ensures chronological order.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame
        df2 (pd.DataFrame): Bitstamp DataFrame

    Returns:
        pd.DataFrame: Concatenated DataFrame with Timestamp as first level
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select the timestamp range
    start, end = 1417411980, 1417417980
    df1 = df1.loc[start:end]
    df2 = df2.loc[start:end]

    # Concatenate with keys
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Swap levels so Timestamp is first, then source
    df = df.swaplevel(0, 1)

    # Sort by Timestamp to ensure chronological order
    return df.sort_index(level=0)
