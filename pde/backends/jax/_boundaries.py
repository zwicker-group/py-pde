"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import numpy as np

from ...backends.numba.utils import make_get_arr_1d
from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    DirichletBC,
    MixedBC,
    NeumannBC,
    _PeriodicBC,
)

if TYPE_CHECKING:
    from .backend import JaxBackend
    from .typing import JaxVirtualPointEvaluator


ResultType = tuple[jax.Array, int, tuple]


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
    # if isinstance(bc, ConstBC2ndOrderBase):
    #     return _make_const2ndorder_virtual_point_evaluator(bc)
    if isinstance(bc, ConstBC1stOrderBase):
        return _make_const1storder_virtual_point_evaluator(bc, backend)
    msg = f"Cannot handle local boundary {bc.__class__}"
    raise NotImplementedError(msg)


def _get_virtual_point_data_1storder(bc: ConstBC1stOrderBase):
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
    """Return function that evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.
        backend (:class`~pde.backends.jax.backend.JaxBackend`):
            The backend that determines where data is moved

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    normal = bc.normal
    axis = bc.axis
    get_arr_1d = make_get_arr_1d(bc.grid.num_axes, bc.axis)

    # calculate necessary constants and move them to device
    const, factor, index = _get_virtual_point_data_1storder(bc)
    const = backend.from_numpy(const)
    factor = backend.from_numpy(factor)

    if bc.homogeneous:

        def virtual_point(arr: jax.Array, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, _ = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const + factor * val_field  # type: ignore

    else:

        def virtual_point(arr: jax.Array, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, bc_idx = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const[bc_idx] + factor[bc_idx] * val_field  # type: ignore

    return virtual_point  # type: ignore
