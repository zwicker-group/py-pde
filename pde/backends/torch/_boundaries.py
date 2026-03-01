"""Defines how boundaries are set using the torch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    ConstBC2ndOrderBase,
    CurvatureBC,
    DirichletBC,
    MixedBC,
    NeumannBC,
    _PeriodicBC,
)
from .utils import TorchOperatorBase

if TYPE_CHECKING:
    from ...grids.boundaries import BoundariesList

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class TorchConstBC1stOrderBoundary(TorchOperatorBase):
    """Class implementing simple first order boundary conditions in torch."""

    const: Tensor
    factor: Tensor

    def __init__(self, bc: ConstBC1stOrderBase, *, dtype: np.dtype):
        super().__init__(dtype=dtype)
        self.bc = bc
        if bc.value_is_linked:
            raise NotImplementedError

        self.i_write = -1 if self.bc.upper else 0
        self.get_virtual_point_data_1storder()

    def get_virtual_point_data_1storder(self):
        """Get the values relevant to calculate the boundary condition."""
        grid = self.bc.grid
        # determine indices into full data array for the data cell near the boundary
        i_lower = 1  # data near lower boundary
        i_upper = grid.shape[self.bc.axis]  # data near upper boundary

        if isinstance(self.bc, _PeriodicBC):
            self.i_read = i_lower if self.bc.upper else i_upper
            const = 0
            factor = -1 if self.bc.flip_sign else 1

        elif isinstance(self.bc, DirichletBC):
            self.i_read = i_upper if self.bc.upper else i_lower
            const = 2 * self.bc.value
            factor = np.full_like(const, -1, dtype=self.dtype)

        elif isinstance(self.bc, NeumannBC):
            dx = grid.discretization[self.bc.axis]
            self.i_read = i_upper if self.bc.upper else i_lower
            const = dx * self.bc.value
            factor = np.ones_like(const, dtype=self.dtype)

        elif isinstance(self.bc, MixedBC):
            dx = grid.discretization[self.bc.axis]
            with np.errstate(invalid="ignore"):
                const = np.asarray(
                    2 * dx * self.bc.const / (2 + dx * self.bc.value), dtype=self.dtype
                )
                factor = np.asarray(
                    (2 - dx * self.bc.value) / (2 + dx * self.bc.value),
                    dtype=self.dtype,
                )

            # correct at places of infinite values
            const[~np.isfinite(factor)] = 0
            factor[~np.isfinite(factor)] = -1

            self.i_read = i_upper if self.bc.upper else i_lower
            const = np.array(const, dtype=self.dtype)
            factor = np.array(factor, dtype=self.dtype)

        else:
            msg = f"Unsupported BC {self.bc}"
            raise TypeError(msg)

        # broadcast values to correct shape
        for name, arr in (("const", const), ("factor", factor)):
            arr = np.asarray(arr)
            if self.bc.homogeneous:
                # add spatial axes in the boundary to enable broadcasting
                arr = arr[(...,) + (np.newaxis,) * (grid.num_axes - 1)]
            # make the data available in the kernel
            self.register_array(name, arr)

    def forward(self, data_full: Tensor, args=None) -> Tensor:
        """Set the virtual points at the boundary."""
        num_axes = self.bc.grid.num_axes
        normal = self.bc.normal
        axis = self.bc.axis

        # set local boundary conditions at the right place
        if num_axes == 1:
            if normal:
                val_field = data_full[..., axis, self.i_read]
            else:
                val_field = data_full[..., self.i_read]
            data_full[..., self.i_write] = self.const + self.factor * val_field

        elif num_axes == 2:
            if axis == 0:
                if normal:
                    val_field = data_full[..., axis, self.i_read, :]
                else:
                    val_field = data_full[..., self.i_read, :]
                data_full[..., self.i_write, :] = self.const + self.factor * val_field

            elif axis == 1:
                if normal:
                    val_field = data_full[..., axis, :, self.i_read]
                else:
                    val_field = data_full[..., self.i_read]
                data_full[..., self.i_write] = self.const + self.factor * val_field

        elif num_axes == 3:
            if axis == 0:
                if normal:
                    val_field = data_full[..., axis, self.i_read, :, :]
                else:
                    val_field = data_full[..., self.i_read, :, :]
                data_full[..., self.i_write, :, :] = (
                    self.const + self.factor * val_field
                )

            elif axis == 1:
                if normal:
                    val_field = data_full[..., axis, :, self.i_read, :]
                else:
                    val_field = data_full[..., self.i_read, :]
                data_full[..., self.i_write, :] = self.const + self.factor * val_field

            elif axis == 2:
                if normal:
                    val_field = data_full[..., axis, :, :, self.i_read]
                else:
                    val_field = data_full[..., self.i_read]
                data_full[..., self.i_write] = self.const + self.factor * val_field

        else:
            raise NotImplementedError
        return data_full


class TorchConstBC2ndOrderBoundary(TorchOperatorBase):
    """Class implementing simple second order boundary conditions in torch."""

    const: Tensor
    factor: Tensor

    def __init__(self, bc: ConstBC2ndOrderBase, *, dtype: np.dtype):
        super().__init__(dtype=dtype)
        self.bc = bc
        if bc.value_is_linked:
            raise NotImplementedError

        self.i_write = -1 if self.bc.upper else 0
        self.get_virtual_point_data_2ndorder()

    def get_virtual_point_data_2ndorder(self):
        """Get the values relevant to calculate the boundary condition."""
        grid = self.bc.grid

        if isinstance(self.bc, CurvatureBC):
            size = self.bc.grid.shape[self.bc.axis]
            dx = self.bc.grid.discretization[self.bc.axis]

            if size < 2:
                msg = "Need at least 2 support points for curvature boundary condition"
                raise RuntimeError(msg)

            value = np.asarray(self.bc.value * dx**2)
            if grid.num_axes == 1:
                self.f1 = 2.0
                self.f2 = -1.0
            elif grid.num_axes == 2:
                self.register_array("f1", np.atleast_1d(2.0))
                self.register_array("f2", np.atleast_1d(-1.0))
            else:
                raise NotImplementedError

            if self.bc.upper:
                self.i1, self.i2 = -2, -3
            else:
                self.i1, self.i2 = 1, 2
        else:
            msg = f"Unsupported BC {self.bc}"
            raise TypeError(msg)

        # broadcast values to correct shape
        if self.bc.homogeneous:
            # add spatial axes in the boundary to enable broadcasting
            value = value[(...,) + (np.newaxis,) * (grid.num_axes - 1)]
        # make the data available in the kernel
        self.register_array("value", value)

    def forward(self, data_full: Tensor, args=None) -> Tensor:
        """Set the virtual points at the boundary."""
        num_axes = self.bc.grid.num_axes
        normal = self.bc.normal
        axis = self.bc.axis

        # set local boundary conditions at the right place
        if num_axes == 1:
            if normal:
                val1 = data_full[..., axis, self.i1]
                val2 = data_full[..., axis, self.i2]
            else:
                val1 = data_full[..., self.i1]
                val2 = data_full[..., self.i2]

            data_full[..., self.i_write] = self.value + self.f1 * val1 + self.f2 * val2

        elif num_axes == 2:
            if axis == 0:
                if normal:
                    val1 = data_full[..., axis, self.i1, 1:-1]
                    val2 = data_full[..., axis, self.i2, 1:-1]
                else:
                    val1 = data_full[..., self.i1, 1:-1]
                    val2 = data_full[..., self.i2, 1:-1]
                virtual_point = self.value + self.f1 * val1 + self.f2 * val2
                data_full[..., self.i_write, 1:-1] = virtual_point

            elif axis == 1:
                if normal:
                    val1 = data_full[..., axis, 1:-1, self.i1]
                    val2 = data_full[..., axis, 1:-1, self.i2]
                else:
                    val1 = data_full[..., 1:-1, self.i1]
                    val2 = data_full[..., 1:-1, self.i2]
                virtual_point = self.value + self.f1 * val1 + self.f2 * val2
                data_full[..., 1:-1, self.i_write] = virtual_point

        else:
            raise NotImplementedError
        return data_full


def make_local_ghost_cell_setter(bc: BCBase, *, dtype: np.dtype) -> torch.nn.Module:
    """Return function that sets virtual points for a local BC.

    Args:
        bc (:class:`~pde.grids.boundaries.local.BCBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.
        dtype:
            The dtype of the data

    Returns:
        function: A function that takes the full data array
    """
    # if isinstance(bc, UserBC):
    #     return _make_user_virtual_point_evaluator(bc)
    if isinstance(bc, ConstBC1stOrderBase):
        return TorchConstBC1stOrderBoundary(bc, dtype=dtype)
    if isinstance(bc, ConstBC2ndOrderBase):
        return TorchConstBC2ndOrderBoundary(bc, dtype=dtype)
    msg = f"Cannot handle local boundary {bc.__class__}"
    raise NotImplementedError(msg)


class GhostCellSetter(torch.nn.Module):
    """Return function that sets virtual points for a local BC."""

    def __init__(self, bcs: BoundariesList, *, dtype: np.dtype):
        """
        Args:
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced and it is assumed that the operator is applied
                to the full field.
            dtype:
                The dtype of the data
        """
        super().__init__()
        self.ghost_cell_setters = torch.nn.ModuleList(
            [
                make_local_ghost_cell_setter(bc_local, dtype=dtype)
                for bc_axis in bcs
                for bc_local in bc_axis
            ]
        )

    def forward(self, data_full: Tensor, args=None) -> Tensor:
        """Set the virtual points at all boundaries."""
        for set_ghost_cells in self.ghost_cell_setters:
            set_ghost_cells(data_full, args=args)
        return data_full
