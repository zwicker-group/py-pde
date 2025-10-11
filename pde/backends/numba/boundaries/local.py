"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Callable

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

from ....grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    ConstBC2ndOrderBase,
    CurvatureBC,
    DirichletBC,
    ExpressionBC,
    MixedBC,
    NeumannBC,
    UserBC,
    _PeriodicBC,
)
from ....tools.numba import address_as_void_pointer, jit, numba_dict
from ....tools.typing import (
    GhostCellSetter,
    NumberOrArray,
    NumericArray,
    VirtualPointEvaluator,
)
from ..utils import make_get_arr_1d


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
            Defines the boundary conditions for a particular side, for which the virtual
            point evaluator should be defined.

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
            Defines the boundary conditions for a particular side, for which the virtual
            point evaluator should be defined.

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


def _make_value_getter(bc: ConstBC1stOrderBase) -> Callable[[], NumericArray]:
    """Return a compiled function for obtaining the value.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC1stOrderBase`):
            Defines the boundary conditions for a particular side, for which the value
            getter should be defined.

    Note:
        This should only be used in numba compiled functions that need to
        support boundary values that can be changed after the function has
        been compiled. In essence, the helper function created here serves
        to get around the compile-time constants that are otherwise created.

    Warning:
        The returned function has a hard-coded reference to the memory
        address of the value error, which must thus be maintained in memory.
        If the address of bc.value changes, a new function needs to be
        created by calling this factory function again.
    """
    # obtain details about the array
    mem_addr = bc.value.ctypes.data
    shape = bc.value.shape
    dtype = bc.value.dtype

    # Note that we tried using register_jitable here, but this lead to
    # problems with address_as_void_pointer

    @nb.njit(nb.typeof(bc._value)(), inline="always")
    def get_value() -> NumericArray:
        """Helper function returning the linked array."""
        return nb.carray(address_as_void_pointer(mem_addr), shape, dtype)  # type: ignore

    # keep a reference to the array to prevent garbage collection
    get_value._value_ref = bc._value

    return get_value  # type: ignore


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

        @register_jitable(inline="always")
        def const_func():
            return const

        @register_jitable(inline="always")
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
            value_func = _make_value_getter(bc)

            @register_jitable(inline="always")
            def const_func():
                return 2 * value_func()

        else:

            @register_jitable(inline="always")
            def const_func():
                return const

        @register_jitable(inline="always")
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
            value_func = _make_value_getter(bc)

            @register_jitable(inline="always")
            def const_func():
                return dx * value_func()

        else:

            @register_jitable(inline="always")
            def const_func():
                return const

        @register_jitable(inline="always")
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
            const_val = np.array(bc.const)
            value_func = _make_value_getter(bc)

            @register_jitable(inline="always")
            def const_func():
                value = value_func()
                const = np.empty_like(value)
                for i in range(value.size):
                    val = value.flat[i]
                    if np.isinf(val):
                        const.flat[i] = 0
                    else:
                        const.flat[i] = 2 * dx * const_val / (2 + dx * val)
                return const

            @register_jitable(inline="always")
            def factor_func():
                value = value_func()
                factor = np.empty_like(value)
                for i in range(value.size):
                    val = value.flat[i]
                    if np.isinf(val):
                        factor.flat[i] = -1
                    else:
                        factor.flat[i] = (2 - dx * val) / (2 + dx * val)
                return factor

        else:
            const = np.array(const)
            factor = np.array(factor)

            @register_jitable(inline="always")
            def const_func():
                return const

            @register_jitable(inline="always")
            def factor_func():
                return factor
    else:
        raise TypeError(f"Unsupported BC {bc}")

    return (const_func, factor_func, index)


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
    const, factor, index = _get_virtual_point_data_1storder(bc)

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


def _get_virtual_point_data_2ndorder(bc: ConstBC2ndOrderBase):
    """Return data suitable for calculating virtual points.

    Args:
        bc (:class:`~pde.grids.boundaries.local.ConstBC2ndOrderBase`):
            Defines the boundary conditions for a particular side, for which the virtual
            point data getter should be defined.

    Returns:
        tuple: the data structure associated with this virtual point
    """
    if isinstance(bc, CurvatureBC):
        size = bc.grid.shape[bc.axis]
        dx = bc.grid.discretization[bc.axis]

        if size < 2:
            raise RuntimeError(
                "Need at least 2 support points to use curvature boundary condition"
            )

        value = np.asarray(bc.value * dx**2)
        f1 = np.full_like(value, 2.0)
        f2 = np.full_like(value, -1.0)
        if bc.upper:
            i1, i2 = size - 1, size - 2
        else:
            i1, i2 = 0, 1
        return (value, f1, i1, f2, i2)

    else:
        raise TypeError(f"Unsupported BC {bc}")


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
    data = _get_virtual_point_data_2ndorder(bc)

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
