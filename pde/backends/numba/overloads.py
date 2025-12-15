"""Defines functions overloads, so numba can use them.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
from numba.experimental import jitclass
from numba.extending import overload

from ...tools.math import OnlineStatistics as OnlineStatistics_np


@nb.vectorize()
def _heaviside_implementation_ufunc(x1, x2):
    """Ufunc implementation of the Heaviside function used for numba and sympy.

    Args:
        x1 (float): Argument of the function
        x2 (float): Value returned when the argument is zero

    Returns:
        float: 0 if x1 is negative, 1 if x1 is positive, and x2 if x1 == 0
    """
    if np.isnan(x1):
        return math.nan
    if x1 == 0:
        return x2
    if x1 < 0:
        return 0.0
    return 1.0


def _heaviside_implementation(x1, x2):
    """Normal implementation of the Heaviside function used for numba and sympy.

    Args:
        x1 (float): Argument of the function
        x2 (float): Value returned when the argument is zero

    Returns:
        float: 0 if x1 is negative, 1 if x1 is positive, and x2 if x1 == 0
    """
    # this extra function is necessary since the `overload` wrapper cannot deal properly
    # with the `_DUFunc` returned by `vectorize`
    return _heaviside_implementation_ufunc(x1, x2)


@overload(np.heaviside)
def np_heaviside(x1, x2):
    """Numba implementation of the Heaviside function."""
    return _heaviside_implementation


# make the `OnlineStatistics` class usable from numba
OnlineStatistics = jitclass(
    [
        ("min", nb.double),
        ("max", nb.double),
        ("mean", nb.double),
        ("_mean2", nb.double),
        ("count", nb.uint),
    ]
)(OnlineStatistics_np)


__all__ = ["OnlineStatistics"]
