"""Common functions that are used by many operators.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Literal

from ....grids.base import GridBase
from ....tools.numba import jit
from ....tools.typing import NumericArray, OperatorType


def make_derivative(
    grid: GridBase,
    axis: int = 0,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
    """Make a derivative operator along a single axis using numba compilation.

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown derivative type `{method}`")

    shape = grid.shape
    num_axes = len(shape)
    dx = grid.discretization[axis]

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if num_axes == 1:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                if method == "central":
                    out[i - 1] = (arr[i + 1] - arr[i - 1]) / (2 * dx)
                elif method == "forward":
                    out[i - 1] = (arr[i + 1] - arr[i]) / dx
                elif method == "backward":
                    out[i - 1] = (arr[i] - arr[i - 1]) / dx

    elif num_axes == 2:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    if method == "central":
                        out[i - 1, j - 1] = (arr_r - arr_l) / (2 * dx)
                    elif method == "forward":
                        out[i - 1, j - 1] = (arr_r - arr[i, j]) / dx
                    elif method == "backward":
                        out[i - 1, j - 1] = (arr[i, j] - arr_l) / dx

    elif num_axes == 3:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        if method == "central":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr_l) / (2 * dx)
                        elif method == "forward":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr[i, j, k]) / dx
                        elif method == "backward":
                            out[i - 1, j - 1, k - 1] = (arr[i, j, k] - arr_l) / dx

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {num_axes:d} axes"
        )

    return diff  # type: ignore


def make_derivative2(grid: GridBase, axis: int = 0) -> OperatorType:
    """Make a second-order derivative operator along a single axis.

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    shape = grid.shape
    num_axes = len(shape)
    scale = 1 / grid.discretization[axis] ** 2

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if num_axes == 1:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * scale

    elif num_axes == 2:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    out[i - 1, j - 1] = (arr_r - 2 * arr[i, j] + arr_l) * scale

    elif num_axes == 3:

        @jit
        def diff(arr: NumericArray, out: NumericArray) -> None:
            """Calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        out[i - 1, j - 1, k - 1] = (
                            arr_r - 2 * arr[i, j, k] + arr_l
                        ) * scale

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {num_axes:d} axes"
        )

    return diff  # type: ignore
