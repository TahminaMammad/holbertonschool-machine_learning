#!/usr/bin/env python3
"""Matrix multiplication module"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    # Check if multiplication is possible
    if len(mat1[0]) != len(mat2):
        return None

    # Result matrix with dimensions (rows of mat1) x (cols of mat2)
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            val = 0
            for k in range(len(mat2)):
                val += mat1[i][k] * mat2[k][j]
            row.append(val)
        result.append(row)
    return result
