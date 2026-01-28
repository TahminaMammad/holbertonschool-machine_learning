#!/usr/bin/env python3
"""
Module that concatenates two pandas DataFrames with indexed timestamps.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames using their Timestamp columns as index.

    - Indexes both DataFrames on the Timestamp column.
    - Selects rows from df2 up to and including timestamp 1417411920.
    - Concatenates the selected df2 rows above df1.
    - Adds keys to distinguish data sources.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to the given timestamp
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    df = pd.concat(
        [df2, df1],
        keys=['bitstamp', 'coinbase']
    )

    return df
