"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ...fields import DataFieldBase, VectorField
from ...solvers import AdaptiveSolverBase, SolverBase
from ...tools.math import OnlineStatistics
from ..base import BackendBase, OperatorInfo, TFunc

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...grids import BoundariesBase, GridBase
    from ...pdes import PDEBase
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import (
        DataSetter,
        GhostCellSetter,
        NumberOrArray,
        NumericArray,
        OperatorType,
        TField,
    )


class NumpyBackend(BackendBase):
    """Basic backend from which all other backends inherit."""

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        return func

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

        def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
            """Default implementation that simply uses the python interface."""
            if args is None:
                boundaries.set_ghost_cells(data_full)
            else:
                boundaries.set_ghost_cells(data_full, *args)

        return ghost_cell_setter

    def make_data_setter(
        self, grid: GridBase, bcs: BoundariesBase | None = None
    ) -> DataSetter:
        """Create a function to set the valid part of a full data array.

        Args:
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                If supplied, the returned function also enforces boundary conditions by
                setting the ghost cells to the correct values
            backend (str):
                The backend to use for making the operator

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        num_axes = grid.num_axes

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
        parameters, like time.

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

        def apply_operator(
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

        return apply_operator

    def make_integrator(
        self, grid: GridBase
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        num_axes = grid.num_axes
        # cell volume varies with position
        cell_volumes = np.broadcast_to(grid.cell_volumes, grid.shape)

        def integrate(arr: NumericArray) -> NumberOrArray:
            """Integrates data over a grid using numpy."""
            amounts = arr * cell_volumes
            return amounts.sum(axis=tuple(range(-num_axes, 0, 1)))  # type: ignore

        return integrate

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
        num_axes = field.grid.num_axes

        def dot(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Numpy implementation to calculate dot product between two fields."""
            rank_a = a.ndim - num_axes
            rank_b = b.ndim - num_axes
            if rank_a < 1 or rank_b < 1:
                msg = "Fields in dot product must have rank >= 1"
                raise TypeError(msg)
            if a.shape[rank_a:] != b.shape[rank_b:]:
                msg = "Shapes of fields are not compatible for dot product"
                raise ValueError(msg)

            if conjugate:
                b = b.conjugate()

            if rank_a == 1 and rank_b == 1:  # result is scalar field
                return np.einsum("i...,i...->...", a, b, out=out)

            if rank_a == 2 and rank_b == 1:  # result is vector field
                return np.einsum("ij...,j...->i...", a, b, out=out)

            if rank_a == 1 and rank_b == 2:  # result is vector field
                return np.einsum("i...,ij...->j...", a, b, out=out)

            if rank_a == 2 and rank_b == 2:  # result is tensor-2 field
                return np.einsum("ij...,jk...->ik...", a, b, out=out)

            msg = f"Unsupported shapes ({a.shape}, {b.shape})"
            raise TypeError(msg)

        return dot

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

        return outer

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
        state = state.copy()  # save this exact state for the closure

        def pde_rhs(state_data: NumericArray, t: float) -> NumericArray:
            """Evaluate the rhs given only a state without the grid."""
            state.data = state_data
            return eq.evolution_rate(state, t).data

        return pde_rhs

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
        if hasattr(eq, "make_noise_realization_numpy"):
            return eq.make_noise_realization_numpy(state)  # type:ignore

        if hasattr(eq, "noise_realization"):
            fields = state.copy()

            def noise_realization(
                state_data: NumericArray, t: float
            ) -> NumericArray | None:
                fields.data = state_data
                noise = eq.noise_realization(fields, t)
                if noise is None:
                    return None
                return noise.data  # type: ignore

            return noise_realization

        msg = f"Noise realization is not implemented for {eq}"
        raise NotImplementedError(msg)

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
        solver.info["dt_statistics"] = OnlineStatistics()

        assert solver.backend == self.name
        if stepper_style == "fixed":
            return solver._make_fixed_stepper(state, dt)
        if stepper_style == "adaptive":
            assert isinstance(solver, AdaptiveSolverBase)
            return solver._make_adaptive_stepper(state)
        msg = f"Numpy backend cannot handle stepper style {stepper_style}"
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

        class NumpyArrayPrinter(PythonCodePrinter):
            """Special sympy printer returning numpy arrays."""

            def _print_ImmutableDenseNDimArray(self, arr):
                arrays = ", ".join(f"asarray({self._print(expr)})" for expr in arr)
                return f"array(broadcast_arrays({arrays}))"

        printer = NumpyArrayPrinter(
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

            def result(*args):
                return func(*args, *const_values)

        else:
            result = func
        return result
