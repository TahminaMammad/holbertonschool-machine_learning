#!/usr/bin/env python3
"""DeepNeuralNetwork with save and load functionality"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    # ... [all previous methods like __init__, forward_prop, train, etc.] ...

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        If filename doesn't end with .pkl, it is appended.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from filename.
        Returns the object, or None if filename doesn't exist.
        """
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
