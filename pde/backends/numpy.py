"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base import BackendBase, OperatorInfo

if TYPE_CHECKING:
    from ..fields.datafield_base import DataFieldBase
    from ..grids.base import GridBase
    from ..grids.boundaries.axes import BoundariesBase
    from ..tools.typing import DataSetter, GhostCellSetter, NumericArray


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
            arr_full[(..., *grid._idx_valid)] = arr
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
