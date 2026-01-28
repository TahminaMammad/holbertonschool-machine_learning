#!/usr/bin/env python3
"""
Visualizes Bitcoin price data after cleaning and aggregation.
"""

import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file


def visualize(df):
    """
    Transforms and visualizes the DataFrame.

    Args:
        df (pd.DataFrame): Raw Bitcoin DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame before plotting.
    """
    # Remove Weighted_Price
    df = df.drop(columns=['Weighted_Price'])

    # Rename Timestamp to Date
    df = df.rename(columns={'Timestamp': 'Date'})

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # Index on Date
    df = df.set_index('Date')

    # Fill missing Close values with previous row
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill Open, High, Low with Close value
    for col in ['Open', 'High', 'Low']:
        df[col] = df[col].fillna(df['Close'])

    # Fill volume columns with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # Filter from 2017 onwards
    df = df['2017-01-01':]

    # Resample daily and aggregate
    df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # Plot
    df.plot(subplots=True, figsize=(12, 10))
    plt.tight_layout()
    plt.show()

    return df


if __name__ == '__main__':
    df = from_file(
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
        ','
    )

    df = visualize(df)
    print(df)
