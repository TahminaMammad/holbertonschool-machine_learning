#!/usr/bin/env python3
"""
Module 9-fill
Provides a function to clean and fill missing values in a DataFrame.
"""


def fill(df):
    """
    Cleans and fills missing values in a DataFrame.

    Steps:
        - Removes the Weighted_Price column.
        - Fills missing values in the Close column with the previous row’s value.
        - Fills missing values in High, Low, and Open columns with the
          corresponding Close value in the same row.
        - Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Remove Weighted_Price column
    if "Weighted_Price" in df.columns:
    df = df.drop(columns=["Weighted_Price"])

    # Fill missing Close values with previous row’s value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill missing High, Low, Open with corresponding Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df
