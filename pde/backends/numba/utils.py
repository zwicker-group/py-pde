"""Defines utilities for the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from numba.extending import register_jitable

from ...tools.typing import NumericArray

if TYPE_CHECKING:
    from ...grids.base import GridBase


def make_get_arr_1d(
    dim: int, axis: int
) -> Callable[[NumericArray, tuple[int, ...]], tuple[NumericArray, int, tuple]]:
    """Create function that extracts a 1d array at a given position.

    Args:
        dim (int):
            The dimension of the space, i.e., the number of axes in the supplied array
        axis (int):
            The axis that is returned as the 1d array

    Returns:
        function: A numba compiled function that takes the full array `arr` and
        an index `idx` (a tuple of `dim` integers) specifying the point where
        the 1d array is extract. The function returns a tuple (arr_1d, i, bc_i),
        where `arr_1d` is the 1d array, `i` is the index `i` into this array
        marking the current point and `bc_i` are the remaining components of
        `idx`, which locate the point in the orthogonal directions.
        Consequently, `i = idx[axis]` and `arr[..., idx] == arr_1d[..., i]`.
    """
    assert 0 <= axis < dim
    ResultType = tuple[NumericArray, int, tuple]

    # extract the correct indices
    if dim == 1:

        def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
            """Extract the 1d array along axis at point idx."""
            i = idx[0]
            bc_idx: tuple = (...,)
            arr_1d = arr
            return arr_1d, i, bc_idx

    elif dim == 2:
        if axis == 0:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y = idx
                bc_idx = (..., y)
                arr_1d = arr[..., :, y]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i = idx
                bc_idx = (..., x)
                arr_1d = arr[..., x, :]
                return arr_1d, i, bc_idx

    elif dim == 3:
        if axis == 0:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y, z = idx
                bc_idx = (..., y, z)
                arr_1d = arr[..., :, y, z]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i, z = idx
                bc_idx = (..., x, z)
                arr_1d = arr[..., x, :, z]
                return arr_1d, i, bc_idx

        elif axis == 2:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, y, i = idx
                bc_idx = (..., x, y)
                arr_1d = arr[..., x, y, :]
                return arr_1d, i, bc_idx

    else:
        raise NotImplementedError

    return register_jitable(inline="always")(get_arr_1d)  # type: ignore


def get_grid_numba_type(grid: GridBase, rank: int = 0):
    """Return numba type corresponding to a particular grid.

    Args:
        grid (GridBase):
            The grid for which we determine the type
        rank (int):
            The rank of the data stored in the grid

    Returns:
        _type_: _description_
    """
    dim = grid.num_axes + rank
    return "f8[" + ", ".join([":"] * dim) + "]"
