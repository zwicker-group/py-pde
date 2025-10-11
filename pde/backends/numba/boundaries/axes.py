"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Sequence

from numba.extending import register_jitable

from ....grids.boundaries.axes import BoundariesBase, BoundariesList, BoundariesSetter
from ....tools.numba import jit
from ....tools.typing import GhostCellSetter, NumericArray
from .axis import make_axis_ghost_cell_setter


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
