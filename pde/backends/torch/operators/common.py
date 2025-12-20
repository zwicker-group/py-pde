"""This module implements infrastructure for differential operators using torch.

.. autosummary::
   :nosignatures:

   TorchOperator

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....grids.boundaries import BoundariesList
from .._boundaries import make_local_ghost_cell_setter

if TYPE_CHECKING:
    from torch import Tensor

    from ....grids import GridBase


class TorchOperator(torch.nn.Module):
    """Base class for operators implemented in torch."""

    data_full: Tensor

    def __init__(
        self, grid: GridBase, boundaries: BoundariesList | None, dtype=np.double
    ):
        """Initialize the torch operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced and it is assumed that the operator is applied
                to the full field.
            dtype:
                The data type of the field
        """
        super().__init__()

        # initialize buffer for full data (including ghost cells)
        self.grid = grid
        full_shape = tuple(n + 2 for n in self.grid.shape)
        data_full = torch.empty(full_shape, dtype=dtype)
        self.register_buffer("data_full", data_full)

        if boundaries is None:
            self.apply_bcs = False

        elif isinstance(boundaries, BoundariesList):
            # get the ghost cell setters for all boundaries
            assert grid == boundaries.grid
            self.apply_bcs = True
            self.ghost_cell_setters = []
            for bc_axis in boundaries:
                for bc_local in bc_axis:
                    ghost_cell_setter = make_local_ghost_cell_setter(bc_local)
                    self.ghost_cell_setters.append(ghost_cell_setter)

        else:
            raise NotImplementedError

    def set_valid(self, arr: Tensor) -> None:
        """Set valid data in the internal full array.

        Args:
            arr (:class:`torch.Tensor`):
                The data of the valid grid points
        """
        if self.grid.num_axes == 1:
            self.data_full[..., 1:-1] = arr
        elif self.grid.num_axes == 2:
            self.data_full[..., 1:-1, 1:-1] = arr
        elif self.grid.num_axes == 3:
            self.data_full[..., 1:-1, 1:-1, 1:-1] = arr
        else:
            raise NotImplementedError

    def set_ghost_cells(self):
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
        for set_ghost_cells in self.ghost_cell_setters:
            set_ghost_cells(self.data_full)

    def set_data_with_bcs(self, arr: Tensor) -> None:
        """Fill internal data array with valid data and set ghost cells."""
        self.set_valid(arr)
        self.set_ghost_cells()
