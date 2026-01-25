#!/usr/bin/env python3
"""
Module that loads data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): The file path to load from
        delimiter (str): The column separator

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
