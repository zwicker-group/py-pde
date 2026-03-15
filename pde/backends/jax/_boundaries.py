"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    ConstBC2ndOrderBase,
    CurvatureBC,
    DirichletBC,
    MixedBC,
    NeumannBC,
    _PeriodicBC,
)

if TYPE_CHECKING:
    import jax

    from .backend import JaxBackend
    from .typing import JaxVirtualPointEvaluator


_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


def make_virtual_point_evaluator(
    bc: BCBase, backend: JaxBackend
) -> JaxVirtualPointEvaluator:
    """Return function that evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.BCBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.
        backend (:class`~pde.backends.jax.backend.JaxBackend`):
            The backend that determines where data is moved

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    # if isinstance(bc, UserBC):
    #     return _make_user_virtual_point_evaluator(bc)
    # if isinstance(bc, ExpressionBC):
    #     return _make_expression_virtual_point_evaluator(bc)
    if isinstance(bc, ConstBC1stOrderBase):
        return _make_const1storder_virtual_point_evaluator(bc, backend)
    if isinstance(bc, ConstBC2ndOrderBase):
        return _make_const2ndorder_virtual_point_evaluator(bc, backend)
    msg = f"Cannot handle local boundary {bc.__class__}"
    raise NotImplementedError(msg)


def extract_one_coordinate(
    arr: jax.Array, num_axes: int, axis: int, index: int, normal: bool = False
) -> jax.Array:
    """Extract one boundary-aligned coordinate slice from an array.

    The returned slice is taken along the boundary axis `axis` at position `index`.
    For vector/tensor data with normal components (`normal=True`), the normal component
    corresponding to `axis` is selected first.

    Args:
        arr (:class:`jax.Array`):
            Full data array, including possible leading component axes.
        num_axes (int):
            Number of spatial axes of the grid (1, 2, or 3).
        axis (int):
            Spatial axis normal to the boundary.
        index (int):
            Index along `axis` to extract.
        normal (bool):
            Whether the array contains a leading normal-component axis.

    Returns:
        :class:`jax.Array`:
            The extracted lower-dimensional slice at the given coordinate.
    """
    if normal:
        if num_axes == 1:
            assert axis == 0
            return arr[..., 0, index]
        if num_axes == 2 and axis == 0:
            return arr[..., 0, index, :]
        if num_axes == 2 and axis == 1:
            return arr[..., 1, :, index]
        if num_axes == 3 and axis == 0:
            return arr[..., 0, index, :, :]
        if num_axes == 3 and axis == 1:
            return arr[..., 1, :, index, :]
        if num_axes == 3 and axis == 2:
            return arr[..., 2, :, :, index]
    else:
        if num_axes == 1:
            assert axis == 0
            return arr[..., index]
        if num_axes == 2 and axis == 0:
            return arr[..., index, :]
        if num_axes == 2 and axis == 1:
            return arr[..., :, index]
        if num_axes == 3 and axis == 0:
            return arr[..., index, :, :]
        if num_axes == 3 and axis == 1:
            return arr[..., :, index, :]
        if num_axes == 3 and axis == 2:
            return arr[..., :, :, index]
    raise NotImplementedError


def _get_virtual_point_data_1storder(
    bc: ConstBC1stOrderBase,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return data suitable for calculating virtual points.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the virtual
            point data getter should be defined.

    Returns:
        tuple: the data structure associated with this virtual point
    """
    if bc.value_is_linked:
        raise NotImplementedError

    if isinstance(bc, _PeriodicBC):
        index = 0 if bc.upper else bc.grid.shape[bc.axis] - 1
        const = np.array(0.0)
        factor = np.array(-1.0 if bc.flip_sign else 1.0)

    elif isinstance(bc, DirichletBC):
        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0
        const = np.array(2.0 * bc.value)
        factor = np.full_like(const, -1.0)

    elif isinstance(bc, NeumannBC):
        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0
        dx = bc.grid.discretization[bc.axis]
        const = np.array(dx * bc.value)
        factor = np.ones_like(dx * bc.value)

    elif isinstance(bc, MixedBC):
        index = bc.grid.shape[bc.axis] - 1 if bc.upper else 0
        dx = bc.grid.discretization[bc.axis]
        with np.errstate(invalid="ignore"):
            const = np.asarray(2.0 * dx * bc.const / (2.0 + dx * bc.value))
            factor = np.asarray((2.0 - dx * bc.value) / (2.0 + dx * bc.value))

        # correct at places of infinite values
        const[~np.isfinite(factor)] = 0.0
        factor[~np.isfinite(factor)] = -1.0
        const = np.array(const)
        factor = np.array(factor)

    else:
        msg = f"Unsupported BC {bc}"
        raise TypeError(msg)

    return (const, factor, index)


def _make_const1storder_virtual_point_evaluator(
    bc: ConstBC1stOrderBase, backend: JaxBackend
) -> JaxVirtualPointEvaluator:
    """Return function that evaluates the value of all virtual points along a boundary.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the
            evaluator should be defined.
        backend (:class`~pde.backends.jax.backend.JaxBackend`):
            The backend that determines where data is moved

    Returns:
        function: A function that takes the data array. The result is the data value at
        the corresponding boundary, which is calculated using the boundary condition.
    """
    num_axes = bc.grid.num_axes
    normal = bc.normal
    axis = bc.axis

    # calculate necessary constants and move them to device
    const, factor, index = _get_virtual_point_data_1storder(bc)
    const = backend.from_numpy(bc._match_data_shape(const))
    factor = backend.from_numpy(bc._match_data_shape(factor))

    def virtual_point(arr: jax.Array, idx: tuple[int, ...], args=None) -> jax.Array:
        """Evaluate the virtual point at `idx`"""
        val_field = extract_one_coordinate(
            arr, num_axes=num_axes, axis=axis, index=index, normal=normal
        )
        return const + factor * val_field  # type: ignore

    return virtual_point  # type: ignore


def _get_virtual_point_data_2ndorder(
    bc: ConstBC2ndOrderBase,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Return data suitable for calculating virtual points.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC2ndOrderBase`):
            Defines the boundary conditions for a particular side, for which the virtual
            point data getter should be defined.

    Returns:
        tuple: the data structure associated with this virtual point
    """
    if bc.value_is_linked:
        raise NotImplementedError

    if isinstance(bc, CurvatureBC):
        size = bc.grid.shape[bc.axis]
        dx = bc.grid.discretization[bc.axis]

        if size < 2:
            msg = "Need at least 2 support points to use curvature boundary condition"
            raise RuntimeError(msg)

        value = np.asarray(bc.value * dx**2)
        f1 = np.full_like(value, 2.0)
        f2 = np.full_like(value, -1.0)
        if bc.upper:
            i1, i2 = size - 1, size - 2
        else:
            i1, i2 = 0, 1
        return (value, f1, i1, f2, i2)

    msg = f"Unsupported BC {bc}"
    raise TypeError(msg)


def _make_const2ndorder_virtual_point_evaluator(
    bc: ConstBC2ndOrderBase, backend: JaxBackend
) -> JaxVirtualPointEvaluator:
    """Return function that evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC2ndOrderBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.
        backend (:class`~pde.backends.jax.backend.JaxBackend`):
            The backend that determines where data is moved

    Returns:
        function: A function that takes the data array. The result is the data value at
        the corresponding boundary, which is calculated using the boundary condition.
    """
    num_axes = bc.grid.num_axes
    normal = bc.normal
    axis = bc.axis
    size = bc.grid.shape[bc.axis]

    if size < 2:
        msg = f"Need two support points along axis {bc.axis} to apply conditions"
        raise ValueError(msg)

    # calculate necessary constants
    value, f1, i1, f2, i2 = _get_virtual_point_data_2ndorder(bc)
    value = backend.from_numpy(bc._match_data_shape(value))
    f1 = backend.from_numpy(bc._match_data_shape(f1))
    f2 = backend.from_numpy(bc._match_data_shape(f2))

    def virtual_point(arr: jax.Array, idx: tuple[int, ...], args=None) -> jax.Array:
        """Evaluate the virtual point at `idx`"""
        val1 = extract_one_coordinate(arr, num_axes, axis=axis, index=i1, normal=normal)
        val2 = extract_one_coordinate(arr, num_axes, axis=axis, index=i2, normal=normal)
        return value + f1 * val1 + f2 * val2  # type: ignore

    return virtual_point  # type: ignore
