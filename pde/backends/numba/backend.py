"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools

from ...grids.base import GridBase
from ...grids.boundaries.axes import BoundariesBase
from ...tools.typing import GhostCellSetter
from ..base import BackendBase, OperatorInfo


class NumbaBackend(BackendBase):
    """Defines numba backend."""

    def load_backend(self) -> None:
        import numba

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
        from ._boundaries import make_axes_ghost_cell_setter

        return make_axes_ghost_cell_setter(boundaries)
