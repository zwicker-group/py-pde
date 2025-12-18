"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    DirichletBC,
    MixedBC,
    NeumannBC,
    _PeriodicBC,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...tools.typing import NumberOrArray

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


def make_virtual_point_evaluator(bc: BCBase):
    """Return function that sets evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.BCBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    # if isinstance(bc, UserBC):
    #     return _make_user_virtual_point_evaluator(bc)
    # if isinstance(bc, ConstBC2ndOrderBase):
    #     return _make_const2ndorder_virtual_point_evaluator(bc)
    if isinstance(bc, ConstBC1stOrderBase):
        return _make_const1storder_virtual_point_evaluator(bc)
    msg = f"Cannot handle local boundary {bc.__class__}"
    raise NotImplementedError(msg)


def make_get_arr_1d(
    dim: int, axis: int
) -> Callable[[Tensor, tuple[int, ...]], tuple[Tensor, int, tuple]]:
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
    ResultType = tuple[Tensor, int, tuple]

    # extract the correct indices
    if dim == 1:

        def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
            """Extract the 1d array along axis at point idx."""
            i = idx[0]
            bc_idx: tuple = (...,)
            arr_1d = arr
            return arr_1d, i, bc_idx

    elif dim == 2:
        if axis == 0:

            def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y = idx
                bc_idx = (..., y)
                arr_1d = arr[..., :, y]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i = idx
                bc_idx = (..., x)
                arr_1d = arr[..., x, :]
                return arr_1d, i, bc_idx

    elif dim == 3:
        if axis == 0:

            def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y, z = idx
                bc_idx = (..., y, z)
                arr_1d = arr[..., :, y, z]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i, z = idx
                bc_idx = (..., x, z)
                arr_1d = arr[..., x, :, z]
                return arr_1d, i, bc_idx

        elif axis == 2:

            def get_arr_1d(arr: Tensor, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, y, i = idx
                bc_idx = (..., x, y)
                arr_1d = arr[..., x, y, :]
                return arr_1d, i, bc_idx

    else:
        raise NotImplementedError

    return get_arr_1d


def _get_virtual_point_data_1storder(bc: ConstBC1stOrderBase):
    """Return data suitable for calculating virtual points.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the virtual
            point data getter should be defined.

    Returns:
        tuple: the data structure associated with this virtual point
    """
    if isinstance(bc, _PeriodicBC):
        index = 0 if bc.upper else bc.grid.shape[bc.axis] - 1
        value: NumberOrArray = -1 if bc.flip_sign else 1
        const = np.array(0)
        factor = np.array(value)

        def const_func():
            return const

        def factor_func():
            return factor

    elif isinstance(bc, DirichletBC):
        const = 2 * bc.value
        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0

        # return boundary data such that dynamically calculated values can
        # be used in numba compiled code. This is a work-around since numpy
        # arrays are copied into closures, making them compile-time
        # constants

        const = np.array(const)
        factor = np.full_like(const, -1)

        if bc.value_is_linked:
            raise NotImplementedError

        def const_func():
            return const

        def factor_func():
            return factor

    elif isinstance(bc, NeumannBC):
        dx = bc.grid.discretization[bc.axis]

        const = dx * bc.value
        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0

        # return boundary data such that dynamically calculated values can
        # be used in numba compiled code. This is a work-around since numpy
        # arrays are copied into closures, making them compile-time
        # constants

        const = np.array(const)
        factor = np.ones_like(const)

        if bc.value_is_linked:
            raise NotImplementedError

        def const_func():
            return const

        def factor_func():
            return factor

    elif isinstance(bc, MixedBC):
        dx = bc.grid.discretization[bc.axis]
        with np.errstate(invalid="ignore"):
            const = np.asarray(2 * dx * bc.const / (2 + dx * bc.value))
            factor = np.asarray((2 - dx * bc.value) / (2 + dx * bc.value))

        # correct at places of infinite values
        const[~np.isfinite(factor)] = 0
        factor[~np.isfinite(factor)] = -1

        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0

        # return boundary data such that dynamically calculated values can
        # be used in numba compiled code. This is a work-around since numpy
        # arrays are copied into closures, making them compile-time
        # constants
        if bc.value_is_linked:
            raise NotImplementedError

        const = np.array(const)
        factor = np.array(factor)

        def const_func():
            return const

        def factor_func():
            return factor
    else:
        msg = f"Unsupported BC {bc}"
        raise TypeError(msg)

    return (const_func, factor_func, index)


def _make_const1storder_virtual_point_evaluator(bc: ConstBC1stOrderBase):
    """Return function that sets evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    normal = bc.normal
    axis = bc.axis
    get_arr_1d = make_get_arr_1d(bc.grid.num_axes, bc.axis)

    # calculate necessary constants
    const, factor, index = _get_virtual_point_data_1storder(bc)

    if bc.homogeneous:

        def virtual_point(arr: Tensor, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, _ = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const() + factor() * val_field  # type: ignore

    else:

        def virtual_point(arr: Tensor, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, bc_idx = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const()[bc_idx] + factor()[bc_idx] * val_field  # type: ignore

    return virtual_point
