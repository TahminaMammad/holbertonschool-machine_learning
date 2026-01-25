#!/usr/bin/env python3
"""
Task 17
Provides a function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): list of coefficients where index represents
                     the power of x
        C (int): integration constant (default 0)

    Returns:
        list: coefficients of the integrated polynomial
        None: if poly or C are not valid
    """
    # Validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Validate C
    if not isinstance(C, int):
        return None

    # Start result with integration constant C
    result = [C]

    # Integrate each term
    for i, coef in enumerate(poly):
        integrated_coef = coef / (i + 1)  # divide by new power
        # Convert to integer if possible
        if integrated_coef.is_integer():
            integrated_coef = int(integrated_coef)
        result.append(integrated_coef)

    # Remove trailing zeros to make the list as small as possible
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
