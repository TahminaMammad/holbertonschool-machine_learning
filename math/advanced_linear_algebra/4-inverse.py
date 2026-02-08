#!/usr/bin/env python3
"""
Module that calculates the inverse of a square matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix

    Args:
        matrix (list of lists): square matrix

    Returns:
        int/float: determinant of the matrix
    """
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]

    det = 0
    for col in range(len(matrix)):
        sub_matrix = []
        for row in matrix[1:]:
            sub_matrix.append(row[:col] + row[col + 1:])
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix

    Args:
        matrix (list of lists): square matrix

    Returns:
        list of lists: minor matrix
    """
    if len(matrix) == 1:
        return [[1]]

    size = len(matrix)
    minor_matrix = []

    for i in range(size):
        row_minor = []
        for j in range(size):
            sub_matrix = []
            for r in range(size):
                if r != i:
                    sub_matrix.append(matrix[r][:j] + matrix[r][j + 1:])
            row_minor.append(determinant(sub_matrix))
        minor_matrix.append(row_minor)
    return minor_matrix


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix

    Args:
        matrix (list of lists): square matrix

    Returns:
        list of lists: cofactor matrix
    """
    if len(matrix) == 1:
        return [[1]]

    minor_matrix = minor(matrix)
    size = len(matrix)
    cofactor_matrix = []

    for i in range(size):
        row = []
        for j in range(size):
            row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix

    Args:
        matrix (list of lists): square matrix

    Returns:
        list of lists: adjugate matrix
    """
    cofactor_matrix = cofactor(matrix)
    size = len(matrix)
    adj = [[cofactor_matrix[j][i] for j in range(size)] for i in range(size)]
    return adj


def inverse(matrix):
    """
    Calculates the inverse of a square matrix

    Args:
        matrix (list of lists): square matrix

    Returns:
        list of lists: inverse of matrix, or None if singular
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    size = len(matrix)
    inv = [[adj[i][j] / det for j in range(size)] for i in range(size)]
    return inv
