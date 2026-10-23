import numpy as np


def sci_notation_latex(value, pos=None):
    """Format a number for a LaTeX-rendered scientific-notation label."""
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10**exponent
    if abs(coefficient - 1.0) < 1e-3:
        return r"10$^{{{}}}$".format(exponent)
    if abs(coefficient - 10.0) < 1e-3:
        return r"10$^{{{}}}$".format(exponent + 1)
    return r"${:.1f} \times 10^{{{}}}$".format(coefficient, exponent)