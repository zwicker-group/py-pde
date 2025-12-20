"""Defines how boundaries are set using the numba backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from ...grids.boundaries.local import (
    BCBase,
    ConstBC1stOrderBase,
    DirichletBC,
    MixedBC,
    NeumannBC,
    _PeriodicBC,
)

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class TorchConstBC1stOrderBoundary(torch.nn.Module):
    """Base class for local boundaries implemented in torch."""

    const: Tensor
    factor: Tensor

    def __init__(self, bc: ConstBC1stOrderBase):
        super().__init__()
        self.bc = bc
        if bc.value_is_linked:
            raise NotImplementedError

        self.i_write = -1 if self.bc.upper else 0
        self.get_virtual_point_data_1storder()

    def get_virtual_point_data_1storder(self):
        """Get the values relevant to calculate the boundary condition."""
        # determine indices into full data array for the data cell near the boundary
        i_lower = 1  # data near lower boundary
        i_upper = self.bc.grid.shape[self.bc.axis]  # data near upper boundary

        if isinstance(self.bc, _PeriodicBC):
            self.i_read = i_lower if self.bc.upper else i_upper
            const = 0
            factor = -1 if self.bc.flip_sign else 1

        elif isinstance(self.bc, DirichletBC):
            self.i_read = i_upper if self.bc.upper else i_lower
            const = 2 * self.bc.value
            factor = np.full_like(const, -1)

        elif isinstance(self.bc, NeumannBC):
            dx = self.bc.grid.discretization[self.bc.axis]
            self.i_read = i_upper if self.bc.upper else i_lower
            const = dx * self.bc.value
            factor = np.ones_like(const)

        elif isinstance(self.bc, MixedBC):
            dx = self.bc.grid.discretization[self.bc.axis]
            with np.errstate(invalid="ignore"):
                const = np.asarray(2 * dx * self.bc.const / (2 + dx * self.bc.value))
                factor = np.asarray((2 - dx * self.bc.value) / (2 + dx * self.bc.value))

            # correct at places of infinite values
            const[~np.isfinite(factor)] = 0
            factor[~np.isfinite(factor)] = -1

            self.i_read = i_upper if self.bc.upper else i_lower
            const = np.array(const)
            factor = np.array(factor)

        else:
            msg = f"Unsupported BC {self.bc}"
            raise TypeError(msg)

        # make the data available in the kernel
        self.register_buffer("const", torch.from_numpy(np.asarray(const)))
        self.register_buffer("factor", torch.from_numpy(np.asarray(factor)))

    def forward(self, data_full: Tensor) -> Tensor:
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
                    val_field = data_full[..., :, self.i_read]
                data_full[..., :, self.i_write] = self.const + self.factor * val_field

        else:
            raise NotImplementedError
        return data_full


def make_local_ghost_cell_setter(bc: BCBase):
    """Return function that sets virtual points for a local BC.

    Args:
        bc (:class:`~pde.grids.boundaries.local.BCBase`):
            Defines the boundary conditions for a particular side, for which the setter
            should be defined.

    Returns:
        function: A function that takes the full data array
    """
    # if isinstance(bc, UserBC):
    #     return _make_user_virtual_point_evaluator(bc)
    # if isinstance(bc, ConstBC2ndOrderBase):
    #     return _make_const2ndorder_virtual_point_evaluator(bc)
    if isinstance(bc, ConstBC1stOrderBase):
        return TorchConstBC1stOrderBoundary(bc)
    msg = f"Cannot handle local boundary {bc.__class__}"
    raise NotImplementedError(msg)
