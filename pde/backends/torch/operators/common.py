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
    rank_in: int = 0
    """int: The rank of the input tensor"""

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the torch operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced and it is assumed that the operator is applied
                to the full field.
            dtype:
                The data type of the field using the numpy convention
        """
        super().__init__()

        # initialize buffer for full data (including ghost cells)
        self.dtype = dtype
        self.grid = grid
        full_shape = (grid.dim,) * self.rank_in + tuple(n + 2 for n in self.grid.shape)
        data_full = np.empty(full_shape, dtype=dtype)
        self.register_array("data_full", data_full)

        if bcs is None:
            self.apply_bcs = False

        elif isinstance(bcs, BoundariesList):
            # get the ghost cell setters for all boundaries
            if grid != bcs.grid:
                msg = "Different grids for operator and BCs"
                raise ValueError(msg)
            self.apply_bcs = True
            self.ghost_cell_setters = [
                make_local_ghost_cell_setter(bc_local, dtype=dtype)
                for bc_axis in bcs
                for bc_local in bc_axis
            ]

        else:
            raise NotImplementedError

    def register_array(self, name: str, arr: np.ndarray | torch.Tensor) -> None:
        """Register an array as a buffer in the torch module.

        Args:
            name (str):
                The name under which the buffer is registered
            arr (:class:`numpy.ndarray` or :class:`torch.Tensor`):
                The array to register. If a numpy array is provided, it will be
                converted to a torch tensor with the appropriate dtype.
        """
        if isinstance(arr, np.ndarray):
            tensor = torch.from_numpy(arr.astype(self.dtype))
        elif isinstance(arr, torch.Tensor):
            tensor = arr
        else:
            raise TypeError

        self.register_buffer(name, tensor)

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

    def set_ghost_cells(self, args=None):
        """Return function that sets the ghost cells on a full array.

        Args:
            args:
                Additional arguments that might be used in the ghost cell setters, e.g.,
                the time `t` during solving a PDE.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        for set_ghost_cells in self.ghost_cell_setters:
            set_ghost_cells(self.data_full, args=args)

    def get_full_data(self, arr: Tensor, args=None) -> Tensor:
        """Get full data array including ghost cells.

        Args:
            arr (:class:`torch.Tensor`):
                The input data. If boundary conditions are applied, this should contain
                only the valid grid points. Otherwise, it should already include ghost
                cells.
            args:
                Additional arguments passed to ghost cell setters, e.g., the time `t`.

        Returns:
            :class:`torch.Tensor`:
                The full data array including ghost cells with boundary conditions
                applied if necessary.
        """
        if self.apply_bcs:
            # `arr` only contains the valid data and we need to apply boundary
            # conditions. We thus use the internal data `self.data_full`
            self.set_valid(arr)
            self.set_ghost_cells(args=args)
            return self.data_full

        # Assume `arr` already contains the full data
        return arr


class IntegralOperator(torch.nn.Module):
    """Operator integrating a field implemented in torch."""

    def __init__(self, grid: GridBase, *, dtype=np.double):
        """Initialize the torch operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            dtype:
                The data type of the field
        """
        super().__init__()

        # initialize cell volumes array necessary for integration
        self.grid = grid
        self.spatial_dims = tuple(range(-grid.num_axes, 0, 1))
        cell_volumes = np.broadcast_to(grid.cell_volumes, grid.shape)
        self.register_array("cell_volumes", cell_volumes)

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        amounts = arr * self.cell_volumes  # type: ignore
        return torch.sum(amounts, dim=self.spatial_dims)
