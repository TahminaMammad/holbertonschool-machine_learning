#!/usr/bin/env python3
"""
Module that cleans a DataFrame by removing Weighted_Price
and filling missing values appropriately.
"""

import pandas as pd


def fill(df):
    """
    Cleans the DataFrame:
    - Removes the Weighted_Price column
    - Fills missing Close values with the previous row’s value
    - Fills missing High, Low, and Open values with the corresponding Close value
    - Sets missing values in Volume_(BTC) and Volume_(Currency) to 0

    Args:
        df (pd.DataFrame): DataFrame containing cryptocurrency data

    Returns:
        pd.DataFrame: Modified DataFrame
    """
    # Remove Weighted_Price column
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # Fill Close NaN with previous row’s value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill High, Low, Open NaN with corresponding Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # Set missing values in Volume columns to 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
