"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Number

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

from ...grids.boundaries.axes import BoundariesBase, BoundariesList, BoundariesSetter
from ...grids.boundaries.axis import BoundaryAxisBase
from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    ConstBC2ndOrderBase,
    ExpressionBC,
    UserBC,
)
from ...tools.numba import jit, numba_dict
from ...tools.typing import GhostCellSetter, NumericArray, VirtualPointEvaluator
from .utils import make_get_arr_1d


def make_axes_ghost_cell_setter(boundaries: BoundariesBase) -> GhostCellSetter:
    """Return function that sets the ghost cells on a full array.

    Args:
        boundaries (:class:`~pde.grids.boundaries.axes.BoundariesBase`):
            Defines the boundary conditions for a particular grid, for which the setter
            should be defined.

    Returns:
        Callable with signature :code:`(data_full: NumericArray, args=None)`, which
        sets the ghost cells of the full data, potentially using additional
        information in `args` (e.g., the time `t` during solving a PDE)
    """
    if isinstance(boundaries, BoundariesList):
        ghost_cell_setters = tuple(
            make_axis_ghost_cell_setter(bc_axis) for bc_axis in boundaries
        )

        # TODO: use numba.literal_unroll
        # # get the setters for all axes
        #
        # from pde.tools.numba import jit
        #
        # @jit
        # def set_ghost_cells(data_full: NumericArray, args=None) -> None:
        #     for f in nb.literal_unroll(ghost_cell_setters):
        #         f(data_full, args=args)
        #
        # return set_ghost_cells

        def chain(
            fs: Sequence[GhostCellSetter], inner: GhostCellSetter | None = None
        ) -> GhostCellSetter:
            """Helper function composing setters of all axes recursively."""

            first, rest = fs[0], fs[1:]

            if inner is None:

                @register_jitable
                def wrap(data_full: NumericArray, args=None) -> None:
                    first(data_full, args=args)

            else:

                @register_jitable
                def wrap(data_full: NumericArray, args=None) -> None:
                    inner(data_full, args=args)
                    first(data_full, args=args)

            if rest:
                return chain(rest, wrap)
            else:
                return wrap  # type: ignore

        return chain(ghost_cell_setters)

    elif isinstance(boundaries, BoundariesSetter):
        return jit(boundaries._setter)  # type: ignore

    else:
        raise NotImplementedError("Cannot handle boundaries {boundaries.__class__}")


def make_axis_ghost_cell_setter(bc_axis: BoundaryAxisBase) -> GhostCellSetter:
    """Return function that sets the ghost cells for a particular axis.

    Args:
        bc_axis (:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`):
            Defines the boundary conditions for a particular axis, for which the setter
            should be defined.

    Returns:
        Callable with signature :code:`(data_full: NumericArray, args=None)`, which
        sets the ghost cells of the full data, potentially using additional
        information in `args` (e.g., the time `t` during solving a PDE)
    """
    # get the functions that handle the data
    ghost_cell_sender_low = make_local_ghost_cell_setter(bc_axis.low)
    ghost_cell_sender_high = make_local_ghost_cell_setter(bc_axis.high)
    ghost_cell_setter_low = make_local_ghost_cell_setter(bc_axis.low)
    ghost_cell_setter_high = make_local_ghost_cell_setter(bc_axis.high)

    @register_jitable
    def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
        """Helper function setting the conditions on all axes."""
        # send boundary information to other nodes if using MPI
        ghost_cell_sender_low(data_full, args=args)
        ghost_cell_sender_high(data_full, args=args)
        # set the actual ghost cells
        ghost_cell_setter_high(data_full, args=args)
        ghost_cell_setter_low(data_full, args=args)

    return ghost_cell_setter  # type: ignore


def make_local_ghost_cell_setter(bc: BCBase) -> GhostCellSetter:
    """Return function that sets the ghost cells for a particular side of an axis.

    Args:
        bc (:class:`~pde.grids.boundaries.local.BCBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        Callable with signature :code:`(data_full: NumericArray, args=None)`, which
        sets the ghost cells of the full data, potentially using additional
        information in `args` (e.g., the time `t` during solving a PDE)
    """
    normal = bc.normal
    axis = bc.axis

    # get information of the virtual points (ghost cells)
    vp_idx = bc.grid.shape[bc.axis] + 1 if bc.upper else 0
    np_idx = bc._get_value_cell_index(with_ghost_cells=False)
    vp_value = make_virtual_point_evaluator(bc)

    if bc.grid.num_axes == 1:  # 1d grid

        @register_jitable
        def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
            """Helper function setting the conditions on all axes."""
            data_valid = data_full[..., 1:-1]
            val = vp_value(data_valid, (np_idx,), args=args)
            if normal:
                data_full[..., axis, vp_idx] = val
            else:
                data_full[..., vp_idx] = val

    elif bc.grid.num_axes == 2:  # 2d grid
        if bc.axis == 0:
            num_y = bc.grid.shape[1]

            @register_jitable
            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1, 1:-1]
                for j in range(num_y):
                    val = vp_value(data_valid, (np_idx, j), args=args)
                    if normal:
                        data_full[..., axis, vp_idx, j + 1] = val
                    else:
                        data_full[..., vp_idx, j + 1] = val

        elif bc.axis == 1:
            num_x = bc.grid.shape[0]

            @register_jitable
            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1, 1:-1]
                for i in range(num_x):
                    val = vp_value(data_valid, (i, np_idx), args=args)
                    if normal:
                        data_full[..., axis, i + 1, vp_idx] = val
                    else:
                        data_full[..., i + 1, vp_idx] = val

    elif bc.grid.num_axes == 3:  # 3d grid
        if bc.axis == 0:
            num_y, num_z = bc.grid.shape[1:]

            @register_jitable
            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                for j in range(num_y):
                    for k in range(num_z):
                        val = vp_value(data_valid, (np_idx, j, k), args=args)
                        if normal:
                            data_full[..., axis, vp_idx, j + 1, k + 1] = val
                        else:
                            data_full[..., vp_idx, j + 1, k + 1] = val

        elif bc.axis == 1:
            num_x, num_z = bc.grid.shape[0], bc.grid.shape[2]

            @register_jitable
            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                for i in range(num_x):
                    for k in range(num_z):
                        val = vp_value(data_valid, (i, np_idx, k), args=args)
                        if normal:
                            data_full[..., axis, i + 1, vp_idx, k + 1] = val
                        else:
                            data_full[..., i + 1, vp_idx, k + 1] = val

        elif bc.axis == 2:
            num_x, num_y = bc.grid.shape[:2]

            @register_jitable
            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                for i in range(num_x):
                    for j in range(num_y):
                        val = vp_value(data_valid, (i, j, np_idx), args=args)
                        if normal:
                            data_full[..., axis, i + 1, j + 1, vp_idx] = val
                        else:
                            data_full[..., i + 1, j + 1, vp_idx] = val

    else:
        raise NotImplementedError("Too many axes")

    if isinstance(bc, UserBC):
        # the (pretty uncommon) UserBC needs a special check, which we add here

        @register_jitable
        def ghost_cell_setter_wrapped(data_full: NumericArray, args=None) -> None:
            """Helper function setting the conditions on all axes."""
            if args is None:
                return  # no-op when no specific arguments are given

            if "virtual_point" in args or "value" in args or "derivative" in args:
                # ghost cells will only be set if any of the above keys were supplied
                ghost_cell_setter(data_full, args=args)
            # else: no-op for the default case where BCs are not set by user

        return ghost_cell_setter_wrapped  # type: ignore
    else:
        # the standard case just uses the ghost_cell_setter as defined above
        return ghost_cell_setter  # type: ignore


def make_virtual_point_evaluator(bc: BCBase) -> VirtualPointEvaluator:
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
    if isinstance(bc, UserBC):
        return _make_user_virtual_point_evaluator(bc)
    elif isinstance(bc, ExpressionBC):
        return _make_expression_virtual_point_evaluator(bc)
    elif isinstance(bc, ConstBC2ndOrderBase):
        return _make_const2ndorder_virtual_point_evaluator(bc)
    elif isinstance(bc, ConstBC1stOrderBase):
        return _make_const1storder_virtual_point_evaluator(bc)
    else:
        raise NotImplementedError("Cannot handle local boundary {bc.__class__}")


def _make_user_virtual_point_evaluator(bc: UserBC) -> VirtualPointEvaluator:
    """Return function that sets evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.UserBC`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    get_arr_1d = make_get_arr_1d(bc.grid.num_axes, bc.axis)
    dx = bc.grid.discretization[bc.axis]

    def extract_value(values, arr: NumericArray, idx: tuple[int, ...]):
        """Helper function that extracts the correct value from supplied ones."""
        if isinstance(values, (nb.types.Number, Number)):
            # scalar was supplied => simply return it
            return values
        elif isinstance(arr, (nb.types.Array, np.ndarray)):
            # array was supplied => extract value at current position
            _, _, bc_idx = get_arr_1d(arr, idx)
            return values[bc_idx]
        else:
            raise TypeError("Either a scalar or an array must be supplied")

    @overload(extract_value)
    def ol_extract_value(values, arr: NumericArray, idx: tuple[int, ...]):
        """Helper function that extracts the correct value from supplied ones."""
        if isinstance(values, (nb.types.Number, Number)):
            # scalar was supplied => simply return it
            def impl(values, arr: NumericArray, idx: tuple[int, ...]):
                return values

        elif isinstance(arr, (nb.types.Array, np.ndarray)):
            # array was supplied => extract value at current position

            def impl(values, arr: NumericArray, idx: tuple[int, ...]):
                _, _, bc_idx = get_arr_1d(arr, idx)
                return values[bc_idx]

        else:
            raise TypeError("Either a scalar or an array must be supplied")

        return impl

    @register_jitable
    def virtual_point(arr: NumericArray, idx: tuple[int, ...], args):
        """Evaluate the virtual point at `idx`"""
        if "virtual_point" in args:
            # set the virtual point directly
            return extract_value(args["virtual_point"], arr, idx)

        elif "value" in args:
            # set the value at the boundary
            value = extract_value(args["value"], arr, idx)
            return 2 * value - arr[idx]

        elif "derivative" in args:
            # set the outward derivative at the boundary
            value = extract_value(args["derivative"], arr, idx)
            return dx * value + arr[idx]

        else:
            # no-op for the default case where BCs are not set by user
            return math.nan

    return virtual_point  # type: ignore


def _make_expression_virtual_point_evaluator(bc: ExpressionBC) -> VirtualPointEvaluator:
    """Return function that sets evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ExpressionBC`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    dx = bc.grid.discretization[bc.axis]
    num_axes = bc.grid.num_axes
    get_arr_1d = make_get_arr_1d(num_axes, bc.axis)
    bc_coords = bc.grid._boundary_coordinates(axis=bc.axis, upper=bc.upper)
    bc_coords = np.moveaxis(bc_coords, -1, 0)  # point coordinates to first axis
    assert num_axes <= 3

    if bc._is_func:
        warn_if_time_not_set = False
    else:
        warn_if_time_not_set = bc._func_expression.depends_on("t")
    func = bc._func(do_jit=True)

    @jit
    def virtual_point(arr: NumericArray, idx: tuple[int, ...], args=None) -> float:
        """Evaluate the virtual point at `idx`"""
        _, _, bc_idx = get_arr_1d(arr, idx)
        grid_value = arr[idx]
        coords = bc_coords[bc_idx]

        # extract time for handling time-dependent BCs
        if args is None or "t" not in args:
            if warn_if_time_not_set:
                raise RuntimeError(
                    "Require value for `t` for time-dependent BC. The value must "
                    "be passed explicitly via `args` when calling a differential "
                    "operator."
                )
            t = 0.0
        else:
            t = float(args["t"])

        if num_axes == 1:
            return func(grid_value, dx, coords[0], t)  # type: ignore
        elif num_axes == 2:
            return func(grid_value, dx, coords[0], coords[1], t)  # type: ignore
        elif num_axes == 3:
            return func(grid_value, dx, coords[0], coords[1], coords[2], t)  # type: ignore
        else:
            # cheap way to signal a problem
            return math.nan

    # evaluate the function to force compilation and catch errors early
    virtual_point(np.zeros([3] * num_axes), (0,) * num_axes, numba_dict(t=0.0))

    return virtual_point  # type: ignore


def _make_const1storder_virtual_point_evaluator(
    bc: ConstBC1stOrderBase,
) -> VirtualPointEvaluator:
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
    const, factor, index = bc.get_virtual_point_data(compiled=True)

    if bc.homogeneous:

        @jit
        def virtual_point(arr: NumericArray, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, _ = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const() + factor() * val_field  # type: ignore

    else:

        @jit
        def virtual_point(arr: NumericArray, idx: tuple[int, ...], args=None) -> float:
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, bc_idx = get_arr_1d(arr, idx)
            if normal:
                val_field = arr_1d[..., axis, index]
            else:
                val_field = arr_1d[..., index]
            return const()[bc_idx] + factor()[bc_idx] * val_field  # type: ignore

    return virtual_point  # type: ignore


def _make_const2ndorder_virtual_point_evaluator(
    bc: ConstBC2ndOrderBase,
) -> VirtualPointEvaluator:
    """Return function that sets evaluates the value of a virtual point.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC2ndOrderBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the data array and an index marking the current
        point, which is assumed to be a virtual point. The result is the data value at
        this point, which is calculated using the boundary condition.
    """
    normal = bc.normal
    axis = bc.axis
    size = bc.grid.shape[bc.axis]
    get_arr_1d = make_get_arr_1d(bc.grid.num_axes, bc.axis)

    if size < 2:
        raise ValueError(
            f"Need two support points along axis {bc.axis} to apply conditions"
        )

    # calculate necessary constants
    data = bc.get_virtual_point_data()

    if bc.homogeneous:

        @register_jitable
        def virtual_point(arr: NumericArray, idx: tuple[int, ...], args=None):
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, _ = get_arr_1d(arr, idx)
            if normal:
                val1 = arr_1d[..., axis, data[2]]
                val2 = arr_1d[..., axis, data[4]]
            else:
                val1 = arr_1d[..., data[2]]
                val2 = arr_1d[..., data[4]]
            return data[0] + data[1] * val1 + data[3] * val2

    else:

        @register_jitable
        def virtual_point(arr: NumericArray, idx: tuple[int, ...], args=None):
            """Evaluate the virtual point at `idx`"""
            arr_1d, _, bc_idx = get_arr_1d(arr, idx)
            if normal:
                val1 = arr_1d[..., axis, data[2]]
                val2 = arr_1d[..., axis, data[4]]
            else:
                val1 = arr_1d[..., data[2]]
                val2 = arr_1d[..., data[4]]
            return data[0][bc_idx] + data[1][bc_idx] * val1 + data[3][bc_idx] * val2

    return virtual_point  # type: ignore
