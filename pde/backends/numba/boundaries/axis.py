"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba.extending import register_jitable

from .local import make_local_ghost_cell_setter

if TYPE_CHECKING:
    from ....grids.boundaries.axis import BoundaryAxisBase
    from ....tools.typing import GhostCellSetter, NumericArray


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
