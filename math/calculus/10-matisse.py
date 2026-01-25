#!/usr/bin/env python3
"""
Task 10
Provides a function that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): list of coefficients where the index represents
                     the power of x

    Returns:
        list: coefficients of the derivative polynomial
        None: if poly is not valid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Constant polynomial
    if len(poly) == 1:
        return [0]

    derivative = []

    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    # If derivative is all zeros
    for value in derivative:
        if value != 0:
            return derivative

    return [0]
