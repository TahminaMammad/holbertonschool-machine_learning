#!/usr/bin/env python3
"""Flip and Switch module"""
import pandas as pd


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order
    and transposes the sorted DataFrame
    """
    return df.sort_index(ascending=False).transpose()
