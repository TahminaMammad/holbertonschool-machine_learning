#!/usr/bin/env python3
"""
Module that fills missing values in a pandas DataFrame.
"""


def fill(df):
    """
    Cleans and fills missing values in the DataFrame.

    - Removes the Weighted_Price column.
    - Fills missing Close values with the previous row's value.
    - Fills missing Open, High, and Low values with the Close
      value of the same row.
    - Sets missing Volume_(BTC) and Volume_(Currency) values to 0.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Modified DataFrame.
    """
    # Remove Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing Close values with previous value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill Open, High, Low with Close value of the same row
    for col in ['Open', 'High', 'Low']:
        df[col] = df[col].fillna(df['Close'])

    # Fill volume columns with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
