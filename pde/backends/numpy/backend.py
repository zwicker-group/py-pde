"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from ...fields import DataFieldBase, VectorField
from ...grids import BoundariesBase, GridBase
from ...pdes import PDEBase
from ...solvers import AdaptiveSolverBase, SolverBase
from ...tools.typing import DataSetter, GhostCellSetter, NumericArray, TField
from ..base import BackendBase, OperatorInfo


class NumpyBackend(BackendBase):
    """Basic backend from which all other backends inherit."""

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

        else:
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
    ) -> Callable[[NumericArray, NumericArray | None, Any], NumericArray]:
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

        def apply_operator(
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

        return apply_operator

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
                raise TypeError("Fields in dot product must have rank >= 1")
            if a.shape[rank_a:] != b.shape[rank_b:]:
                raise ValueError("Shapes of fields are not compatible for dot product")

            if conjugate:
                b = b.conjugate()

            if rank_a == 1 and rank_b == 1:  # result is scalar field
                return np.einsum("i...,i...->...", a, b, out=out)

            elif rank_a == 2 and rank_b == 1:  # result is vector field
                return np.einsum("ij...,j...->i...", a, b, out=out)

            elif rank_a == 1 and rank_b == 2:  # result is vector field
                return np.einsum("i...,ij...->j...", a, b, out=out)

            elif rank_a == 2 and rank_b == 2:  # result is tensor-2 field
                return np.einsum("ij...,jk...->ik...", a, b, out=out)

            else:
                raise TypeError(f"Unsupported shapes ({a.shape}, {b.shape})")

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
            raise TypeError("Can only define outer product between vector fields")

        def outer(
            a: NumericArray, b: NumericArray, out: NumericArray | None = None
        ) -> NumericArray:
            """Calculate the outer product using numpy."""
            return np.einsum("i...,j...->ij...", a, b, out=out)

        return outer

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
        state = state.copy()  # save this exact state for the closure

        def pde_rhs(state_data: NumericArray, t: float) -> NumericArray:
            """Evaluate the rhs given only a state without the grid."""
            state.data = state_data
            return eq.evolution_rate(state, t, **kwargs).data

        pde_rhs._backend = "numpy"  # type: ignore
        return pde_rhs

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
        if stepper_style == "fixed":
            return solver._make_fixed_stepper(state, dt)
        elif stepper_style == "adaptive":
            assert isinstance(solver, AdaptiveSolverBase)
            return solver._make_adaptive_stepper(state)
        else:
            raise NotImplementedError(
                f"Numpy backend cannot handle stepper style {stepper_style}"
            )
