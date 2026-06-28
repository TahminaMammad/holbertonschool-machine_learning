#!/usr/bin/env python3
"""
Preprocess Bitcoin datasets for time series forecasting.

This script:
- Loads the Coinbase and Bitstamp datasets.
- Merges them.
- Removes duplicate timestamps.
- Sorts by timestamp.
- Fills missing values.
- Normalizes numerical features.
- Saves the processed dataset.
- Saves the scaler for future use.
"""

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price"
]


def load_dataset(filename):
    """
    Load a CSV dataset.

    Args:
        filename (str): CSV file path.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filename)


def merge_datasets(coinbase, bitstamp):
    """
    Merge two datasets.

    Args:
        coinbase (DataFrame): Coinbase data.
        bitstamp (DataFrame): Bitstamp data.

    Returns:
        DataFrame: Merged data.
    """
    merged = pd.concat([coinbase, bitstamp], ignore_index=True)

    merged = merged.sort_values("Timestamp")

    merged = merged.drop_duplicates(
        subset="Timestamp",
        keep="first"
    )

    return merged


def clean_data(data):
    """
    Clean missing values.

    Args:
        data (DataFrame): Raw data.

    Returns:
        DataFrame: Cleaned data.
    """
    data = data.ffill()
    data = data.bfill()

    data = data.dropna()

    return data


def normalize(data):
    """
    Normalize selected features.

    Args:
        data (DataFrame): Dataset.

    Returns:
        tuple:
            normalized DataFrame,
            fitted scaler
    """
    scaler = MinMaxScaler()

    data[FEATURES] = scaler.fit_transform(data[FEATURES])

    return data, scaler


def main():
    """
    Execute preprocessing.
    """
    coinbase = load_dataset("coinbase.csv")
    bitstamp = load_dataset("bitstamp.csv")

    data = merge_datasets(
        coinbase,
        bitstamp
    )

    data = clean_data(data)

    data, scaler = normalize(data)

    data.to_csv(
        "btc_processed.csv",
        index=False
    )

    joblib.dump(
        scaler,
        "scaler.pkl"
    )

    print("Preprocessing completed.")
    print("Saved: btc_processed.csv")
    print("Saved: scaler.pkl")


if __name__ == "__main__":
    main()
