#!/usr/bin/env python3
"""
Module for calculating the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix whose determinant is to be calculated.

    Returns:
        int or float: The determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    # Validate input type
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle empty matrix case
    if matrix == [[]]:
        return 1

    # Ensure matrix is square
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: Laplace expansion
    det = 0
    for col in range(size):
        # Build minor matrix
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det
