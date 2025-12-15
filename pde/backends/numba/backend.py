"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import numba as nb
import numpy as np
from numba.extending import is_jitted, register_jitable
from numba.extending import overload as nb_overload

from ...fields import DataFieldBase, VectorField
from ...grids import DimensionError, DomainError, GridBase
from ...grids.boundaries.axes import BoundariesBase, BoundariesList, BoundariesSetter
from ...grids.boundaries.local import BCBase, UserBC
from ...solvers import AdaptiveSolverBase, SolverBase
from ..numpy.backend import NumpyBackend, OperatorInfo
from . import grids
from .overloads import OnlineStatistics
from .utils import get_common_numba_dtype, jit, make_array_constructor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...grids.boundaries.axis import BoundaryAxisBase
    from ...pdes import PDEBase
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import (
        DataSetter,
        FloatingArray,
        GhostCellSetter,
        Number,
        NumberOrArray,
        NumericArray,
        OperatorType,
        TField,
    )
    from ..base import TFunc


class NumbaBackend(NumpyBackend):
    """Defines numba backend."""

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        return jit(func)  # type: ignore

    def get_registered_operators(self, grid_id: GridBase | type[GridBase]) -> set[str]:
        """Returns all operators defined for a grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase` or its type):
                Grid for which the operator need to be returned
        """
        operators = super().get_registered_operators(grid_id)

        # add operators calculating derivate along a coordinate
        for ax in getattr(grid_id, "axes", []):
            operators |= {
                f"d_d{ax}",
                f"d_d{ax}_forward",
                f"d_d{ax}_backward",
                f"d2_d{ax}2",
            }

        return operators

    def get_operator_info(
        self, grid: GridBase, operator: str | OperatorInfo
    ) -> OperatorInfo:
        """Return the operator defined on this grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.

        Returns:
            :class:`~pde.grids.base.OperatorInfo`: information for the operator
        """
        if isinstance(operator, OperatorInfo):
            return operator
        assert isinstance(operator, str)

        try:
            # try the default method for determining operators
            return super().get_operator_info(grid, operator)

        except NotImplementedError:
            # deal with some special patterns that are often used
            if operator.startswith("d_d"):
                # create a special operator that takes a first derivative along one axis
                from .operators.common import make_derivative

                # determine axis to which operator is applied (and the method to use)
                axis_name = operator[len("d_d") :]
                for direction in ["central", "forward", "backward"]:
                    if axis_name.endswith("_" + direction):
                        method = direction
                        axis_name = axis_name[: -len("_" + direction)]
                        break
                else:
                    method = "central"

                axis_id = grid.axes.index(axis_name)
                factory = functools.partial(
                    make_derivative,
                    axis=axis_id,
                    method=method,  # type: ignore
                )
                return OperatorInfo(factory, rank_in=0, rank_out=0, name=operator)

            if operator.startswith("d2_d") and operator.endswith("2"):
                # create a special operator taking a second derivative along one axis
                from .operators.common import make_derivative2

                axis_id = grid.axes.index(operator[len("d2_d") : -1])
                factory = functools.partial(make_derivative2, axis=axis_id)
                return OperatorInfo(factory, rank_in=0, rank_out=0, name=operator)

        # throw an informative error since operator was not found
        op_list = ", ".join(sorted(self.get_registered_operators(grid)))
        msg = (
            f"'{operator}' is not one of the defined operators ({op_list}). Custom "
            "operators can be added using the `register_operator` method."
        )
        raise NotImplementedError(msg)

    def _make_local_ghost_cell_setter(self, bc: BCBase) -> GhostCellSetter:
        """Return function that sets the ghost cells for a particular side of an axis.

        Args:
            bc (:class:`~pde.grids.boundaries.local.BCBase`):
                Defines the boundary conditions for a particular side, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        from ._boundaries import make_virtual_point_evaluator

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
            msg = "Too many axes"
            raise NotImplementedError(msg)

        if isinstance(bc, UserBC):
            # the (pretty uncommon) UserBC needs a special check, which we add here

            @register_jitable
            def ghost_cell_setter_wrapped(data_full: NumericArray, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                if args is None:
                    return  # no-op when no specific arguments are given

                if "virtual_point" in args or "value" in args or "derivative" in args:
                    # ghost cells will only be set if any of the above keys are supplied
                    ghost_cell_setter(data_full, args=args)
                # else: no-op for the default case where BCs are not set by user

            return ghost_cell_setter_wrapped  # type: ignore
        # the standard case just uses the ghost_cell_setter as defined above
        return ghost_cell_setter  # type: ignore

    def _make_axis_ghost_cell_setter(
        self, bc_axis: BoundaryAxisBase
    ) -> GhostCellSetter:
        """Return function that sets the ghost cells for a particular axis.

        Args:
            bc_axis (:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`):
                Defines the boundary conditions for a particular axis, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        # get the functions that handle the data
        # ghost_cell_sender_low = make_local_ghost_cell_sender(bc_axis.low)
        # ghost_cell_sender_high = make_local_ghost_cell_sender(bc_axis.high)
        ghost_cell_setter_low = self._make_local_ghost_cell_setter(bc_axis.low)
        ghost_cell_setter_high = self._make_local_ghost_cell_setter(bc_axis.high)

        @register_jitable
        def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
            """Helper function setting the conditions on all axes."""
            # send boundary information to other nodes if using MPI
            # ghost_cell_sender_low(data_full, args=args)
            # ghost_cell_sender_high(data_full, args=args)
            # set the actual ghost cells
            ghost_cell_setter_high(data_full, args=args)
            ghost_cell_setter_low(data_full, args=args)

        return ghost_cell_setter  # type: ignore

    def make_ghost_cell_setter(self, boundaries: BoundariesBase) -> GhostCellSetter:
        """Return function that sets the ghost cells on a full array.

        Args:
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesBase`):
                Defines the boundary conditions for a particular grid, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        if isinstance(boundaries, BoundariesList):
            ghost_cell_setters = tuple(
                self._make_axis_ghost_cell_setter(bc_axis) for bc_axis in boundaries
            )

            # TODO: use numba.literal_unroll
            # # get the setters for all axes
            #
            # from numba import jit
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
                return wrap  # type: ignore

            return chain(ghost_cell_setters)

        if isinstance(boundaries, BoundariesSetter):
            return jit(boundaries._setter)  # type: ignore

        msg = "Cannot handle boundaries {boundaries.__class__}"
        raise NotImplementedError(msg)

    def make_data_setter(
        self, grid: GridBase, bcs: BoundariesBase | None = None
    ) -> DataSetter:
        """Create a function to set the valid part of a full data array.

        Args:
            grid
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesBase`):
                Defines the boundary conditions for a particular grid, for which the
                setter should be defined.

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        num_axes = grid.num_axes

        @jit
        def set_valid(
            data_full: NumericArray, data_valid: NumericArray, args=None
        ) -> None:
            """Set valid part of the data (without ghost cells)

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The full array with ghost cells that the data is written to
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to `data_full`
            """
            if num_axes == 1:
                data_full[..., 1:-1] = data_valid
            elif num_axes == 2:
                data_full[..., 1:-1, 1:-1] = data_valid
            elif num_axes == 3:
                data_full[..., 1:-1, 1:-1, 1:-1] = data_valid
            else:
                raise NotImplementedError

        if bcs is None:
            # just set the valid elements and leave ghost cells with arbitrary values
            return set_valid  # type: ignore

        # set the valid elements and the ghost cells according to boundary condition
        set_bcs = self.make_ghost_cell_setter(bcs)

        @jit
        def set_valid_bcs(
            data_full: NumericArray, data_valid: NumericArray, args=None
        ) -> None:
            """Set valid part of the data and the ghost cells using BCs.

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The full array with ghost cells that the data is written to
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to `data_full`
                args (dict):
                    Extra arguments affecting the boundary conditions
            """
            set_valid(data_full, data_valid)
            set_bcs(data_full, args=args)

        return set_valid_bcs  # type: ignore

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesBase,
        **kwargs,
    ) -> OperatorType:
        """Return a compiled function applying an operator with boundary conditions.

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                The boundary conditions used before the operator is applied
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array.

        The function also accepts an optional parameter `args`, which is forwarded to
        `set_ghost_cells`. This allows setting boundary conditions based on external
        parameters, like time. Note that since the returned operator will always be
        compiled by Numba, the arguments need to be compatible with Numba. The
        following example shows how to pass the current time `t`:

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        operator_raw = operator_info.factory(grid, **kwargs)

        # calculate shapes of the full data
        shape_in_valid = (grid.dim,) * operator_info.rank_in + grid.shape
        shape_in_full = (grid.dim,) * operator_info.rank_in + grid._shape_full
        shape_out = (grid.dim,) * operator_info.rank_out + grid.shape

        # define numpy version of the operator
        def apply_op(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            # check input array
            if arr.shape != shape_in_valid:
                msg = f"Incompatible shapes {arr.shape} != {shape_in_valid}"
                raise ValueError(msg)

            # ensure `out` array is allocated and has the right shape
            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            elif out.shape != shape_out:
                msg = f"Incompatible shapes {out.shape} != {shape_out}"
                raise ValueError(msg)

            # prepare input with boundary conditions
            arr_full = np.empty(shape_in_full, dtype=arr.dtype)
            arr_full[(..., *grid._idx_valid)] = arr  # type: ignore
            bcs.set_ghost_cells(arr_full, args=args)

            # apply operator
            operator_raw(arr_full, out)

            # return valid part of the output
            return out

        # overload `apply_op` with numba-compiled version
        set_valid_w_bc = grid._make_set_valid(bcs=bcs)

        if not is_jitted(operator_raw):
            operator_raw = jit(operator_raw)

        @nb_overload(apply_op, inline="always")
        def apply_op_ol(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Make numba implementation of the operator."""
            if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                # need to allocate memory for `out`

                def apply_op_impl(
                    arr: NumericArray, out: NumericArray | None = None, args=None
                ) -> NumericArray:
                    """Allocates `out` and applies operator to the data."""
                    if arr.shape != shape_in_valid:
                        raise ValueError("Incompatible shapes of input array")  # noqa: EM101, TRY003

                    out = np.empty(shape_out, dtype=arr.dtype)
                    # prepare input with boundary conditions
                    arr_full = np.empty(shape_in_full, dtype=arr.dtype)
                    set_valid_w_bc(arr_full, arr, args=args)  # type: ignore

                    # apply operator
                    operator_raw(arr_full, out)

                    # return valid part of the output
                    return out

            else:
                # reuse provided `out` array

                def apply_op_impl(
                    arr: NumericArray, out: NumericArray | None = None, args=None
                ) -> NumericArray:
                    """Applies operator to the data without allocating out."""
                    if TYPE_CHECKING:
                        assert isinstance(out, np.ndarray)  # help type checker
                    if arr.shape != shape_in_valid:
                        raise ValueError("Incompatible shapes of input array")  # noqa: EM101, TRY003
                    if out.shape != shape_out:
                        raise ValueError("Incompatible shapes of output array")  # noqa: EM101, TRY003

                    # prepare input with boundary conditions
                    arr_full = np.empty(shape_in_full, dtype=arr.dtype)
                    set_valid_w_bc(arr_full, arr, args=args)  # type: ignore

                    # apply operator
                    operator_raw(arr_full, out)

                    # return valid part of the output
                    return out

            return apply_op_impl  # type: ignore

        @jit
        def apply_op_compiled(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            return apply_op(arr, out, args)

        # return the compiled versions of the operator
        return apply_op_compiled  # type: ignore

    def _make_local_integrator(
        self, grid: GridBase
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        num_axes = grid.num_axes
        # cell volume varies with position
        get_cell_volume = grids.make_cell_volume_getter(grid=grid, flat_index=True)

        def integrate_local(arr: NumericArray) -> NumberOrArray:
            """Integrates data over a grid using numpy."""
            # Dummy function so we can overwrite it using numba. This function will only
            # be called when the numba backend is used with DISABLE_JIT=True
            amounts = arr * grid.cell_volumes
            return amounts.sum(axis=tuple(range(-num_axes, 0, 1)))  # type: ignore

        # We need to overload the integrate function since we want to be able to
        # integrate scalar and tensorial fields, which lead to different signatures.

        @nb_overload(integrate_local)
        def ol_integrate_local(
            arr: NumericArray,
        ) -> Callable[[NumericArray], NumberOrArray]:
            """Integrates data over a grid using numba."""
            if arr.ndim == num_axes:
                # `arr` is a scalar field
                grid_shape = grid.shape

                def impl(arr: NumericArray) -> Number:
                    """Integrate a scalar field."""
                    assert arr.shape == grid_shape
                    total = 0
                    for i in range(arr.size):
                        total += get_cell_volume(i) * arr.flat[i]
                    return total

            else:
                # `arr` is a tensorial field with rank >= 1
                tensor_shape = (grid.dim,) * (arr.ndim - num_axes)
                data_shape = tensor_shape + grid.shape

                def impl(arr: NumericArray) -> NumericArray:  # type: ignore
                    """Integrate a tensorial field."""
                    assert arr.shape == data_shape
                    total = np.zeros(tensor_shape)
                    for idx in np.ndindex(*tensor_shape):
                        arr_comp = arr[idx]
                        for i in range(arr_comp.size):
                            total[idx] += get_cell_volume(i) * arr_comp.flat[i]
                    return total

            return impl

        return integrate_local

    def make_integrator(
        self, grid: GridBase
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        integrate_local = self._make_local_integrator(grid)

        @jit
        def integrate_global(arr: NumericArray) -> NumberOrArray:
            """Integrate data.

            Args:
                arr (:class:`~numpy.ndarray`): discretized data on grid
            """
            return integrate_local(arr)

        return integrate_global  # type: ignore

    def make_inner_prod_operator(
        self, field: DataFieldBase, *, conjugate: bool = True
    ) -> Callable[[NumericArray, NumericArray, NumericArray | None], NumericArray]:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the inner product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        dot = super().make_inner_prod_operator(field, conjugate=conjugate)

        dim = field.grid.dim
        num_axes = field.grid.num_axes

        @register_jitable
        def maybe_conj(arr: NumericArray) -> NumericArray:
            """Helper function implementing optional conjugation."""
            return arr.conjugate() if conjugate else arr

        def get_rank(arr: nb.types.Type | nb.types.Optional) -> int:
            """Determine rank of field with type `arr`"""
            arr_typ = arr.type if isinstance(arr, nb.types.Optional) else arr
            if not isinstance(arr_typ, (np.ndarray, nb.types.Array)):
                msg = f"Dot argument must be array, not  {arr_typ.__class__}"
                raise nb.errors.TypingError(msg)
            rank = arr_typ.ndim - num_axes
            if rank < 1:
                msg = (
                    f"Rank={rank} too small for dot product. Use a normal product "
                    "instead."
                )
                raise nb.NumbaTypeError(msg)
            return rank

        @nb_overload(dot, inline="always")
        def dot_ol(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Numba implementation to calculate dot product between two fields."""
            # get (and check) rank of the input arrays
            rank_a = get_rank(a)
            rank_b = get_rank(b)

            if rank_a == 1 and rank_b == 1:  # result is scalar field

                @register_jitable
                def calc(a: NumericArray, b: NumericArray, out: NumericArray) -> None:
                    out[:] = a[0] * maybe_conj(b[0])
                    for j in range(1, dim):
                        out[:] += a[j] * maybe_conj(b[j])

            elif rank_a == 2 and rank_b == 1:  # result is vector field

                @register_jitable
                def calc(a: NumericArray, b: NumericArray, out: NumericArray) -> None:
                    for i in range(dim):
                        out[i] = a[i, 0] * maybe_conj(b[0])
                        for j in range(1, dim):
                            out[i] += a[i, j] * maybe_conj(b[j])

            elif rank_a == 1 and rank_b == 2:  # result is vector field

                @register_jitable
                def calc(a: NumericArray, b: NumericArray, out: NumericArray) -> None:
                    for i in range(dim):
                        out[i] = a[0] * maybe_conj(b[0, i])
                        for j in range(1, dim):
                            out[i] += a[j] * maybe_conj(b[j, i])

            elif rank_a == 2 and rank_b == 2:  # result is tensor-2 field

                @register_jitable
                def calc(a: NumericArray, b: NumericArray, out: NumericArray) -> None:
                    for i in range(dim):
                        for j in range(dim):
                            out[i, j] = a[i, 0] * maybe_conj(b[0, j])
                            for k in range(1, dim):
                                out[i, j] += a[i, k] * maybe_conj(b[k, j])

            else:
                msg = "Inner product for these ranks"
                raise NotImplementedError(msg)

            if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                # function is called without `out` -> allocate memory
                rank_out = rank_a + rank_b - 2
                a_shape = (dim,) * rank_a + field.grid.shape
                b_shape = (dim,) * rank_b + field.grid.shape
                out_shape = (dim,) * rank_out + field.grid.shape
                dtype = get_common_numba_dtype(a, b)

                def dot_impl(
                    a: NumericArray,
                    b: NumericArray,
                    out: NumericArray | None = None,
                ) -> NumericArray:
                    """Helper function allocating output array."""
                    assert a.shape == a_shape
                    assert b.shape == b_shape
                    out = np.empty(out_shape, dtype=dtype)
                    calc(a, b, out)
                    return out

            else:
                # function is called with `out` argument -> reuse `out` array

                def dot_impl(
                    a: NumericArray,
                    b: NumericArray,
                    out: NumericArray | None = None,
                ) -> NumericArray:
                    """Helper function without allocating output array."""
                    assert a.shape == a_shape
                    assert b.shape == b_shape
                    assert out.shape == out_shape  # type: ignore
                    calc(a, b, out)
                    return out  # type: ignore

            return dot_impl  # type: ignore

        @jit
        def dot_compiled(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Numba implementation to calculate dot product between two fields."""
            return dot(a, b, out)

        return dot_compiled  # type: ignore

    def make_outer_prod_operator(
        self, field: DataFieldBase
    ) -> Callable[[NumericArray, NumericArray, NumericArray | None], NumericArray]:
        """Return operator calculating the outer product between two fields.

        This supports typically only supports products between two vector fields.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the outer product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        if not isinstance(field, VectorField):
            msg = "Can only define outer product between vector fields"
            raise TypeError(msg)

        def outer(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Calculate the outer product using numpy."""
            return np.einsum("i...,j...->ij...", a, b, out=out)

        # overload `outer` with a numba-compiled version

        dim = field.grid.dim
        num_axes = field.grid.num_axes

        def check_rank(arr: nb.types.Type | nb.types.Optional) -> None:
            """Determine rank of field with type `arr`"""
            arr_typ = arr.type if isinstance(arr, nb.types.Optional) else arr
            if not isinstance(arr_typ, (np.ndarray, nb.types.Array)):
                msg = f"Arguments must be array, not  {arr_typ.__class__}"
                raise nb.errors.TypingError(msg)
            assert arr_typ.ndim == 1 + num_axes

        # create the inner function calculating the outer product
        @register_jitable
        def calc(a: NumericArray, b: NumericArray, out: NumericArray) -> NumericArray:
            """Calculate outer product between fields `a` and `b`"""
            for i in range(dim):
                for j in range(dim):
                    out[i, j, :] = a[i] * b[j]
            return out

        @nb_overload(outer, inline="always")
        def outer_ol(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Numba implementation to calculate outer product between two fields."""
            # get (and check) rank of the input arrays
            check_rank(a)
            check_rank(b)
            in_shape = (dim, *field.grid.shape)
            out_shape = (dim, dim, *field.grid.shape)

            if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                # function is called without `out` -> allocate memory
                dtype = get_common_numba_dtype(a, b)

                def outer_impl(
                    a: NumericArray,
                    b: NumericArray,
                    out: NumericArray | None = None,
                ) -> NumericArray:
                    """Helper function allocating output array."""
                    assert a.shape == b.shape == in_shape
                    out = np.empty(out_shape, dtype=dtype)
                    calc(a, b, out)
                    return out

            else:
                # function is called with `out` argument -> reuse `out` array

                def outer_impl(
                    a: NumericArray,
                    b: NumericArray,
                    out: NumericArray | None = None,
                ) -> NumericArray:
                    """Helper function without allocating output array."""
                    # check input
                    assert a.shape == b.shape == in_shape
                    assert out.shape == out_shape  # type: ignore
                    calc(a, b, out)
                    return out  # type: ignore

            return outer_impl  # type: ignore

        @jit
        def outer_compiled(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Numba implementation to calculate outer product between two fields."""
            return outer(a, b, out)

        return outer_compiled  # type: ignore

    def make_interpolator(
        self,
        field: DataFieldBase,
        *,
        fill: Number | None = None,
        with_ghost_cells: bool = False,
    ) -> Callable[[FloatingArray, NumericArray], NumberOrArray]:
        r"""Returns a function that can be used to interpolate values.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the interpolator is defined
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the ghost points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.

        Returns:
            A function which returns interpolated values when called with arbitrary
            positions within the space of the grid.
        """
        grid = field.grid
        num_axes = field.grid.num_axes
        data_shape = field.data_shape

        # convert `fill` to dtype of data
        if fill is not None:
            if field.rank == 0:
                fill = field.data.dtype.type(fill)  # type: ignore
            else:
                fill = np.broadcast_to(fill, field.data_shape).astype(field.data.dtype)  # type: ignore

        # create the method to interpolate data at a single point
        interpolate_single = grids.make_single_interpolator(
            grid=grid, fill=fill, with_ghost_cells=with_ghost_cells
        )

        # provide a method to access the current data of the field
        if with_ghost_cells:
            get_data_array = make_array_constructor(field._data_full)
        else:
            get_data_array = make_array_constructor(field.data)

        dim_error_msg = f"Dimension of point does not match axes count {num_axes}"

        @jit
        def interpolator(
            point: FloatingArray, data: NumericArray | None = None
        ) -> NumericArray:
            """Return the interpolated value at the position `point`

            Args:
                point (:class:`~numpy.ndarray`):
                    The list of points. This point coordinates should be given along the
                    last axis, i.e., the shape should be `(..., num_axes)`.
                data (:class:`~numpy.ndarray`, optional):
                    The discretized field values. If omitted, the data of the current
                    field is used, which should be the default. However, this option can
                    be useful to interpolate other fields defined on the same grid
                    without recreating the interpolator. If a data array is supplied, it
                    needs to be the full data if `with_ghost_cells == True`, and
                    otherwise only the valid data.

            Returns:
                :class:`~numpy.ndarray`: The interpolated values at the points
            """
            # check input
            point = np.atleast_1d(point)
            if point.shape[-1] != num_axes:
                raise DimensionError(dim_error_msg)
            point_shape = point.shape[:-1]

            if data is None:
                # reconstruct data field from memory address
                data = get_data_array()

            # interpolate at every valid point
            out = np.empty(data_shape + point_shape, dtype=data.dtype)
            for idx in np.ndindex(*point_shape):
                out[(..., *idx)] = interpolate_single(data, point[idx])

            return out

        # store a reference to the data so it is not garbage collected too early
        interpolator._data = field.data

        return interpolator  # type: ignore

    def make_inserter(
        self, grid: GridBase, *, with_ghost_cells: bool = False
    ) -> Callable[[NumericArray, FloatingArray, NumberOrArray], None]:
        """Return a compiled function to insert values at interpolated positions.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the grid points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.

        Returns:
            callable: A function with signature (data, position, amount), where `data`
            is the numpy array containing the field data, position is denotes the
            position in grid coordinates, and `amount` is the  that is to be added to
            the field.
        """
        cell_volume = grids.make_cell_volume_getter(grid=grid, flat_index=False)

        if grid.num_axes == 1:
            # specialize for 1-dimensional interpolation
            data_x = grids.make_interpolation_axis_data(
                grid=grid, axis=0, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: NumericArray, point: FloatingArray, amount: NumberOrArray
            ) -> None:
                """Add an amount to a field at an interpolated position.

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                c_li, c_hi, w_l, w_h = data_x(float(point[0]))

                if c_li == -42:  # out of bounds
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)

                data[..., c_li] += w_l * amount / cell_volume(c_li)
                data[..., c_hi] += w_h * amount / cell_volume(c_hi)

        elif grid.num_axes == 2:
            # specialize for 2-dimensional interpolation
            data_x = grids.make_interpolation_axis_data(
                grid=grid, axis=0, with_ghost_cells=with_ghost_cells
            )
            data_y = grids.make_interpolation_axis_data(
                grid=grid, axis=1, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: NumericArray, point: FloatingArray, amount: NumberOrArray
            ) -> None:
                """Add an amount to a field at an interpolated position.

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(float(point[0]))
                c_yli, c_yhi, w_yl, w_yh = data_y(float(point[1]))

                if c_xli == -42 or c_yli == -42:  # out of bounds
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)

                cell_vol = cell_volume(c_xli, c_yli)
                data[..., c_xli, c_yli] += w_xl * w_yl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yhi)
                data[..., c_xli, c_yhi] += w_xl * w_yh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yli)
                data[..., c_xhi, c_yli] += w_xh * w_yl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yhi)
                data[..., c_xhi, c_yhi] += w_xh * w_yh * amount / cell_vol

        elif grid.num_axes == 3:
            # specialize for 3-dimensional interpolation
            data_x = grids.make_interpolation_axis_data(
                grid=grid, axis=0, with_ghost_cells=with_ghost_cells
            )
            data_y = grids.make_interpolation_axis_data(
                grid=grid, axis=1, with_ghost_cells=with_ghost_cells
            )
            data_z = grids.make_interpolation_axis_data(
                grid=grid, axis=2, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: NumericArray, point: FloatingArray, amount: NumberOrArray
            ) -> None:
                """Add an amount to a field at an interpolated position.

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(float(point[0]))
                c_yli, c_yhi, w_yl, w_yh = data_y(float(point[1]))
                c_zli, c_zhi, w_zl, w_zh = data_z(float(point[2]))

                if c_xli == -42 or c_yli == -42 or c_zli == -42:  # out of bounds
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)

                cell_vol = cell_volume(c_xli, c_yli, c_zli)
                data[..., c_xli, c_yli, c_zli] += w_xl * w_yl * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yli, c_zhi)
                data[..., c_xli, c_yli, c_zhi] += w_xl * w_yl * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xli, c_yhi, c_zli)
                data[..., c_xli, c_yhi, c_zli] += w_xl * w_yh * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yhi, c_zhi)
                data[..., c_xli, c_yhi, c_zhi] += w_xl * w_yh * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yli, c_zli)
                data[..., c_xhi, c_yli, c_zli] += w_xh * w_yl * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yli, c_zhi)
                data[..., c_xhi, c_yli, c_zhi] += w_xh * w_yl * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yhi, c_zli)
                data[..., c_xhi, c_yhi, c_zli] += w_xh * w_yh * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yhi, c_zhi)
                data[..., c_xhi, c_yhi, c_zhi] += w_xh * w_yh * w_zh * amount / cell_vol

        else:
            msg = (
                f"Compiled interpolation not implemented for dimension {grid.num_axes}"
            )
            raise NotImplementedError(msg)

        return insert  # type: ignore

    def make_pde_rhs(
        self, eq: PDEBase, state: TField
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE
        """
        return eq.make_pde_rhs_numba(state)

    def make_noise_realization(
        self, eq: PDEBase, state: TField
    ) -> Callable[[NumericArray, float], NumericArray | None]:
        """Return a function for evaluating the noise term of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function calculating noise
        """
        if hasattr(eq, "make_noise_realization_numba"):
            return eq.make_noise_realization_numba(state)  # type: ignore

        msg = (
            "Noise needs to be implemented by defining the "
            "`make_noise_realization_numba` method for the PDE class."
        )
        raise NotImplementedError(msg)

    def make_expression_function(
        self,
        expression: ExpressionBase,
        *,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
    ) -> Callable[..., NumberOrArray]:
        """Return a function evaluating an expression for a particular backend.

        Args:
            expression (:class:`~pde.tools.expression.ExpressionBase`):
                The expression that is converted to a function
            single_arg (bool):
                Determines whether the returned function accepts all variables in a
                single argument as an array or whether all variables need to be
                supplied separately.
            user_funcs (dict):
                Additional functions that can be used in the expression.

        Returns:
            function: the function
        """
        import sympy
        from sympy.printing.pycode import PythonCodePrinter

        from ...tools.expressions import SPECIAL_FUNCTIONS

        # collect all the user functions
        user_functions = expression.user_funcs.copy()
        if user_funcs is not None:
            user_functions.update(user_funcs)
        user_functions.update(SPECIAL_FUNCTIONS)

        # transform the user functions, so they can be compiled using numba
        def compile_func(func):
            if isinstance(func, np.ufunc):
                # this is a work-around that allows to compile numpy ufuncs
                return jit(lambda *args: func(*args))
            return jit(func)

        user_functions = {k: compile_func(v) for k, v in user_functions.items()}

        # initialize the printer that deals with numpy arrays correctly

        class ListArrayPrinter(PythonCodePrinter):
            """Special sympy printer returning arrays as lists."""

            def _print_ImmutableDenseNDimArray(self, arr):
                arrays = ", ".join(f"{self._print(expr)}" for expr in arr)
                return f"[{arrays}]"

        printer = ListArrayPrinter(
            {
                "fully_qualified_modules": False,
                "inline": True,
                "allow_unknown_functions": True,
                "user_functions": {k: k for k in user_functions},
            }
        )

        # determine the list of variables that the function depends on
        variables = (expression.vars,) if single_arg else tuple(expression.vars)
        constants = tuple(expression.consts)

        # turn the expression into a callable function
        self._logger.info("Parse sympy expression `%s`", expression._sympy_expr)
        func = sympy.lambdify(
            variables + constants,
            expression._sympy_expr,
            modules=[user_functions, "numpy"],
            printer=printer,
        )

        # Apply the constants if there are any. Note that we use this pattern of a
        # partial function instead of replacing the constants in the sympy expression
        # directly since sympy does not work well with numpy arrays.
        if constants:
            const_values = tuple(expression.consts[c] for c in constants)

            func = register_jitable(func)

            def result(*args):
                return func(*args, *const_values)

        else:
            result = func
        return jit(result)  # type: ignore

    def _make_expression_array(
        self, expression: ExpressionBase, *, single_arg: bool = True
    ) -> Callable[[NumericArray, NumericArray | None], NumericArray]:
        """Compile the tensor expression such that a numpy array is returned.

        Args:
            expression (:class:`~pde.tools.expression.ExpressionBase`):
                The expression that is converted to a function
            single_arg (bool):
                Whether the compiled function expects all arguments as a single array
                or whether they are supplied individually.
        """
        import builtins

        import sympy
        from sympy.utilities.lambdify import _get_namespace

        if not isinstance(expression._sympy_expr, sympy.Array):
            msg = "Expression must be an array"
            raise TypeError(msg)
        variables = ", ".join(v for v in expression.vars)
        shape = expression._sympy_expr.shape

        lines = [
            f"    out[{str((*idx, ...))[1:-1]}] = {expr}"
            for idx, expr in np.ndenumerate(expression._sympy_expr)
        ]
        # TODO: replace the np.ndindex with np.ndenumerate eventually. This does not
        # work with numpy 1.18, so we have the work around using np.ndindex

        # TODO: We should also support constants similar to ScalarExpressions. They
        # could be written in separate lines and prepended to the actual code. However,
        # we would need to make sure to print numpy arrays correctly.

        if variables:
            # the expression takes variables as input

            if single_arg:
                # the function takes a single input array
                first_dim = 0 if len(expression.vars) == 1 else 1
                code = "def _generated_function(arr, out=None):\n"
                code += "    arr = asarray(arr)\n"
                code += f"    {variables} = arr\n"
                code += "    if out is None:\n"
                code += f"        out = empty({shape} + arr.shape[{first_dim}:])\n"

            else:
                # the function takes each variables as an argument
                code = f"def _generated_function({variables}, out=None):\n"
                code += "    if out is None:\n"
                code += f"        out = empty({shape} + shape({expression.vars[0]}))\n"

        else:
            # the expression is constant
            if single_arg:
                code = "def _generated_function(arr=None, out=None):\n"
            else:
                code = "def _generated_function(out=None):\n"
            code += "    if out is None:\n"
            code += f"        out = empty({shape})\n"

        code += "\n".join(lines) + "\n"
        code += "    return out"

        self._logger.debug("Code for `get_compiled_array`: %s", code)

        namespace = _get_namespace("numpy")
        namespace["builtins"] = builtins
        namespace.update(expression.user_funcs)
        local_vars: dict[str, Any] = {}
        exec(code, namespace, local_vars)
        function = local_vars["_generated_function"]

        return jit(function)  # type: ignore

    def make_inner_stepper(
        self,
        solver: SolverBase,
        stepper_style: Literal["fixed", "adaptive"],
        state: TField,
        dt: float,
    ) -> Callable:
        """Return a stepper function using an explicit scheme.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step used (Uses :attr:`SolverBase.dt_default` if `None`)

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        assert solver.backend == self.name

        from ._solvers import make_adaptive_stepper, make_fixed_stepper

        solver.info["dt_statistics"] = OnlineStatistics()

        if stepper_style == "fixed":
            return make_fixed_stepper(solver, state, dt=dt)
        if stepper_style == "adaptive":
            assert isinstance(solver, AdaptiveSolverBase)
            return make_adaptive_stepper(solver, state)
        raise NotImplementedError

    def make_mpi_synchronizer(
        self, operator: int | str = "MAX"
    ) -> Callable[[float], float]:
        """Return function that synchronizes values between multiple MPI processes.

        Warning:
            The default implementation does not synchronize anything. This is simply a
            hook, which can be used by backends that support MPI

        Args:
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".

        Returns:
            Function that can be used to synchronize values across nodes
        """
        return register_jitable(super().make_mpi_synchronizer(operator=operator))  # type: ignore
