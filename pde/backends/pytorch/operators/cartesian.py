"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   make_laplace

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....grids import CartesianGrid
from ....grids.boundaries import BoundariesList
from .. import pytorch_backend

if TYPE_CHECKING:
    from torch import Tensor

    from ....grids.boundaries.axis import BoundaryAxisBase
    from ....grids.boundaries.local import BCBase


class TorchOperator(torch.nn.Module):
    def __init__(self, boundaries: BoundariesList, dtype):
        super().__init__()

        # initialize buffer for full data (including ghost cells)
        self.grid = boundaries.grid
        full_shape = tuple(n + 2 for n in self.grid.shape)
        self.data_full = torch.empty(full_shape, dtype=dtype)

        # prepare boundary conditions
        if isinstance(boundaries, BoundariesList):
            # get the setters for all axes
            self.ghost_cell_setters = tuple(
                self._make_axis_ghost_cell_setter(bc_axis) for bc_axis in boundaries
            )
        else:
            raise NotImplementedError

    def _make_local_ghost_cell_setter(self, bc: BCBase):
        """Return function that sets the ghost cells for a particular side of an axis.

        Args:
            bc (:class:`~pde.grids.boundaries.local.BCBase`):
                Defines the boundary conditions for a particular side, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        from .._boundaries import make_virtual_point_evaluator

        normal = bc.normal
        axis = bc.axis

        # get information of the virtual points (ghost cells)
        vp_idx = bc.grid.shape[bc.axis] + 1 if bc.upper else 0
        np_idx = bc._get_value_cell_index(with_ghost_cells=False)
        vp_value = make_virtual_point_evaluator(bc)

        if bc.grid.num_axes == 1:  # 1d grid

            @torch.compile
            def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1]
                val = vp_value(data_valid, (np_idx,), args=args)
                if normal:
                    data_full[..., axis, vp_idx] = val
                else:
                    data_full[..., vp_idx] = val

        elif bc.grid.num_axes == 2:  # 2d grid
            if bc.axis == 0:
                num_y = bc.grid.shape[1]

                @torch.compile
                def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for j in range(num_y):
                        val = vp_value(data_valid, (np_idx, j), args=args)
                        if normal:
                            data_full[..., axis, vp_idx, j + 1] = val
                        else:
                            data_full[..., vp_idx, j + 1] = val

            elif bc.axis == 1:
                num_x = bc.grid.shape[0]

                @torch.compile
                def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for i in range(num_x):
                        val = vp_value(data_valid, (i, np_idx), args=args)
                        if normal:
                            data_full[..., axis, i + 1, vp_idx] = val
                        else:
                            data_full[..., i + 1, vp_idx] = val

        elif bc.grid.num_axes == 3:  # 3d grid
            if bc.axis == 0:
                num_y, num_z = bc.grid.shape[1:]

                @torch.compile
                def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for j in range(num_y):
                        for k in range(num_z):
                            val = vp_value(data_valid, (np_idx, j, k), args=args)
                            if normal:
                                data_full[..., axis, vp_idx, j + 1, k + 1] = val
                            else:
                                data_full[..., vp_idx, j + 1, k + 1] = val

            elif bc.axis == 1:
                num_x, num_z = bc.grid.shape[0], bc.grid.shape[2]

                @torch.compile
                def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for k in range(num_z):
                            val = vp_value(data_valid, (i, np_idx, k), args=args)
                            if normal:
                                data_full[..., axis, i + 1, vp_idx, k + 1] = val
                            else:
                                data_full[..., i + 1, vp_idx, k + 1] = val

            elif bc.axis == 2:
                num_x, num_y = bc.grid.shape[:2]

                @torch.compile
                def ghost_cell_setter(data_full: Tensor, args=None) -> None:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for j in range(num_y):
                            val = vp_value(data_valid, (i, j, np_idx), args=args)
                            if normal:
                                data_full[..., axis, i + 1, j + 1, vp_idx] = val
                            else:
                                data_full[..., i + 1, j + 1, vp_idx] = val

        else:
            msg = "Too many axes"
            raise NotImplementedError(msg)

        # the standard case just uses the ghost_cell_setter as defined above
        return ghost_cell_setter

    def _make_axis_ghost_cell_setter(self, bc_axis: BoundaryAxisBase):
        """Return function that sets the ghost cells for a particular axis.

        Args:
            bc_axis (:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`):
                Defines the boundary conditions for a particular axis, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        # get the functions that handle the data
        ghost_cell_setter_low = self._make_local_ghost_cell_setter(bc_axis.low)
        ghost_cell_setter_high = self._make_local_ghost_cell_setter(bc_axis.high)

        def ghost_cell_setter(data_full: Tensor, args=None) -> None:
            """Helper function setting the conditions on all axes."""
            # set the actual ghost cells
            ghost_cell_setter_high(data_full, args=args)
            ghost_cell_setter_low(data_full, args=args)

        return ghost_cell_setter

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

        def set_ghost_cells(args=None) -> None:
            for set_ghost_cells in self.ghost_cell_setters:
                set_ghost_cells(self.data_full, args=args)

    def set_data_with_bcs(self, arr: Tensor) -> None:
        self.set_valid(arr)
        self.set_ghost_cells()


@pytorch_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
class CartesianLaplacian(TorchOperator):
    def __init__(self, boundaries: BoundariesList, dtype):
        super().__init__(boundaries, dtype)
        self.scale = self.grid.discretization**-2

    def forward(self, arr: Tensor) -> Tensor:
        self.set_data_with_bcs(arr)

        if self.grid.num_axes == 1:
            return (
                self.data_full[:-2] - 2 * self.data_full[1:-1] + self.data_full[2:]
            ) * self.scale  # type: ignore

        if self.grid.num_axes == 2:
            lap_x = (
                self.data_full[:-2, 1:-1]
                - 2 * self.data_full[1:-1, 1:-1]
                + self.data_full[2:, 1:-1]
            ) * self.scale[0]
            lap_y = (
                self.data_full[1:-1, :-2]
                - 2 * self.data_full[1:-1, 1:-1]
                + self.data_full[1:-1, 2:]
            ) * self.scale[1]
            return lap_x + lap_y  # type: ignore

        raise NotImplementedError


__all__ = ["CartesianLaplacian"]
