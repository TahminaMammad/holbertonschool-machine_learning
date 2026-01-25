#!/usr/bin/env python3
"""
Script to clean and visualize cryptocurrency data.
"""

import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")

# Remove Weighted_Price column
if "Weighted_Price" in df.columns:
    df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert timestamp values to datetime
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index the DataFrame on Date
df = df.set_index("Date")

# Fill missing values
df["Close"] = df["Close"].fillna(method="ffill")
for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Filter data from 2017 onwards
df = df.loc["2017-01-01":]

# Resample daily and aggregate
df = df.resample("D").agg(
    {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)

# Plot the Close price
df["Close"].plot(title="Daily Close Price (2017 and beyond)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Return the transformed DataFrame
print(df)
