"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   make_laplace

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....grids import CartesianGrid, GridBase
from ....grids.boundaries import BoundariesList
from .. import torch_backend
from .._boundaries import make_local_ghost_cell_setter

if TYPE_CHECKING:
    from torch import Tensor

    from ..utils import AnyDType


class TorchOperator(torch.nn.Module):
    """Base class for operators implemented in torch."""

    data_full: Tensor

    def __init__(
        self, grid: GridBase, boundaries: BoundariesList | None, dtype=np.double
    ):
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


@torch_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
class CartesianLaplacian(TorchOperator):
    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        dtype: AnyDType = np.double,
    ):
        super().__init__(grid, boundaries, dtype)
        self.scale = self.grid.discretization**-2

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        if self.apply_bcs:
            self.set_data_with_bcs(arr)
            data_full = self.data_full
        else:
            data_full = arr

        if self.grid.num_axes == 1:
            return (data_full[:-2] - 2 * data_full[1:-1] + data_full[2:]) * self.scale  # type: ignore

        if self.grid.num_axes == 2:
            lap_x = (
                data_full[:-2, 1:-1] - 2 * data_full[1:-1, 1:-1] + data_full[2:, 1:-1]
            ) * self.scale[0]
            lap_y = (
                data_full[1:-1, :-2] - 2 * data_full[1:-1, 1:-1] + data_full[1:-1, 2:]
            ) * self.scale[1]
            return lap_x + lap_y  # type: ignore

        raise NotImplementedError


__all__ = ["CartesianLaplacian"]
