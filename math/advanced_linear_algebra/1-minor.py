#!/usr/bin/env python3
"""
Module for calculating the minor matrix of a square matrix.
"""

from 0-determinant import determinant


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose minor matrix is to be
        calculated.

    Returns:
        list of lists: The minor matrix of the given matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    # Validate input type
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix → minor is [[1]]
    if size == 1:
        return [[1]]

    # Build minor matrix
    minors = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            # Build submatrix excluding row i and column j
            submatrix = [r[:j] + r[j + 1:] for k, r in enumerate(matrix)
                         if k != i]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)

    return minors
