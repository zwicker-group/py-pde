"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
from typing import Callable

import numba as nb
import numpy as np
from numba.extending import is_jitted
from numba.extending import overload as nb_overload

from ...grids.base import GridBase
from ...grids.boundaries.axes import BoundariesBase
from ...tools.numba import jit
from ...tools.typing import DataSetter, GhostCellSetter, NumericArray
from ..base import BackendBase, OperatorInfo


class NumbaBackend(BackendBase):
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
