"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   CartesianLaplacian

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....grids import CartesianGrid, GridBase
from .. import torch_backend
from .common import TorchOperator

if TYPE_CHECKING:
    from torch import Tensor

    from ....grids.boundaries import BoundariesList
    from ..utils import AnyDType


@torch_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
class CartesianLaplacian(TorchOperator):
    """Cartesian Laplace using torch."""

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

        if self.grid.num_axes == 3:
            data_mid2 = 2 * data_full[1:-1, 1:-1, 1:-1]
            lap_x = (
                data_full[:-2, 1:-1, 1:-1] - data_mid2 + data_full[2:, 1:-1, 1:-1]
            ) * self.scale[0]
            lap_y = (
                data_full[1:-1, :-2, 1:-1] - data_mid2 + data_full[1:-1, 2:, 1:-1]
            ) * self.scale[1]
            lap_z = (
                data_full[1:-1, 1:-1, :-2] - data_mid2 + data_full[1:-1, 1:-1, 2:]
            ) * self.scale[2]
            return lap_x + lap_y + lap_z  # type: ignore

        raise NotImplementedError


__all__ = ["CartesianLaplacian"]
