#!/usr/bin/env python3
"""
Module that creates a pandas DataFrame from a dictionary.
"""

import pandas as pd


# Create the DataFrame from dictionary
df = pd.DataFrame(
    {"First": [0.0, 0.5, 1.0, 1.5], "Second": ["one", "two", "three", "four"]},
    index=["A", "B", "C", "D"],
)
