"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
from typing import Callable, Literal

import numba as nb
import numpy as np
from numba.extending import is_jitted, register_jitable
from numba.extending import overload as nb_overload

from ...fields import DataFieldBase, VectorField
from ...grids import BoundariesBase, DimensionError, GridBase
from ...pdes import PDEBase
from ...solvers import AdaptiveSolverBase, SolverBase
from ...tools.numba import get_common_numba_dtype, jit, make_array_constructor
from ...tools.typing import (
    DataSetter,
    FloatingArray,
    GhostCellSetter,
    Number,
    NumberOrArray,
    NumericArray,
    TField,
)
from ..numpy.backend import NumpyBackend, OperatorInfo


class NumbaBackend(NumpyBackend):
    """Defines numba backend."""

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

            elif operator.startswith("d2_d") and operator.endswith("2"):
                # create a special operator that takes a second derivative along one axis
                from .operators.common import make_derivative2

                axis_id = grid.axes.index(operator[len("d2_d") : -1])
                factory = functools.partial(make_derivative2, axis=axis_id)
                return OperatorInfo(factory, rank_in=0, rank_out=0, name=operator)

        # throw an informative error since operator was not found
        op_list = ", ".join(sorted(self.get_registered_operators(grid)))
        raise NotImplementedError(
            f"'{operator}' is not one of the defined operators ({op_list}). Custom "
            "operators can be added using the `register_operator` method."
        )

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
        from .boundaries.axes import make_axes_ghost_cell_setter

        return make_axes_ghost_cell_setter(boundaries)

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

        else:
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
    ) -> Callable[..., NumericArray]:
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
                raise ValueError(f"Incompatible shapes {arr.shape} != {shape_in_valid}")

            # ensure `out` array is allocated and has the right shape
            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            elif out.shape != shape_out:
                raise ValueError(f"Incompatible shapes {out.shape} != {shape_out}")

            # prepare input with boundary conditions
            arr_full = np.empty(shape_in_full, dtype=arr.dtype)
            arr_full[(...,) + grid._idx_valid] = arr
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
                        raise ValueError(f"Incompatible shapes of input array")

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
                    assert isinstance(out, np.ndarray)  # help type checker
                    if arr.shape != shape_in_valid:
                        raise ValueError(f"Incompatible shapes of input array")
                    if out.shape != shape_out:
                        raise ValueError(f"Incompatible shapes of output array")

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
                raise nb.errors.TypingError(
                    f"Dot argument must be array, not  {arr_typ.__class__}"
                )
            rank = arr_typ.ndim - num_axes
            if rank < 1:
                raise nb.NumbaTypeError(
                    f"Rank={rank} too small for dot product. Use a normal product "
                    "instead."
                )
            return rank  # type: ignore

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
                raise NotImplementedError("Inner product for these ranks")

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
            raise TypeError("Can only define outer product between vector fields")

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
                raise nb.errors.TypingError(
                    f"Arguments must be array, not  {arr_typ.__class__}"
                )
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
            in_shape = (dim,) + field.grid.shape
            out_shape = (dim, dim) + field.grid.shape

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
                fill = np.broadcast_to(fill, field.data_shape).astype(field.data.dtype)

        # create the method to interpolate data at a single point
        interpolate_single = grid._make_interpolator_compiled(
            fill=fill, with_ghost_cells=with_ghost_cells
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
                out[(...,) + idx] = interpolate_single(data, point[idx])

            return out  # type: ignore

        # store a reference to the data so it is not garbage collected too early
        interpolator._data = field.data

        return interpolator  # type: ignore

    def make_pde_rhs(
        self, eq: PDEBase, state: TField, **kwargs
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
        return eq._make_pde_rhs_numba_cached(state, **kwargs)

    def make_sde_rhs(
        self, eq: PDEBase, state: TField, **kwargs
    ) -> Callable[[NumericArray, float], tuple[NumericArray, NumericArray]]:
        """Return a function for evaluating the right hand side of the SDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE
            together with a noise realization.
        """
        return eq._make_sde_rhs_numba_cached(state, **kwargs)

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

        from .solvers import make_adaptive_stepper, make_fixed_stepper

        if stepper_style == "fixed":
            return make_fixed_stepper(solver, state, dt=dt)
        elif stepper_style == "adaptive":
            assert isinstance(solver, AdaptiveSolverBase)
            return make_adaptive_stepper(solver, state)
        else:
            raise NotImplementedError
