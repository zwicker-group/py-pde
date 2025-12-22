r"""This module implements differential operators on polar grids.

.. autosummary::
   :nosignatures:

   PolarLaplacian
   PolarGradient
   PolarGradientSquared
   PolarDivergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from ....grids import GridBase, PolarSymGrid
from .. import torch_backend
from .common import TorchOperator

if TYPE_CHECKING:
    from torch import Tensor

    from ....grids.boundaries import BoundariesList
    from ..utils import AnyDType


@torch_backend.register_operator(PolarSymGrid, "laplace", rank_in=0, rank_out=0)
class PolarLaplacian(TorchOperator):
    """Polar Laplace using torch."""

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Polar Laplacian operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, bcs, dtype=dtype)

        # calculate preliminary quantities
        dr = grid.discretization[0]
        factor_r = 1 / (2 * self.grid.axes_coords[0] * dr)
        self.register_buffer("factor_r", torch.from_numpy(factor_r))
        self.dr_2 = 1 / dr**2

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        term1 = (data_full[2:] - 2 * data_full[1:-1] + data_full[:-2]) * self.dr_2
        term2 = (data_full[2:] - data_full[:-2]) * self.factor_r  # type: ignore
        return term1 + term2  # type: ignore


@torch_backend.register_operator(PolarSymGrid, "gradient", rank_in=0, rank_out=1)
class PolarGradient(TorchOperator):
    """Polar gradient operator using torch."""

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        """Initialize the Polar gradient operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
            method (str):
                The method for calculating the derivative. Possible values are
                'central', 'forward', and 'backward'.
        """
        super().__init__(grid, bcs, dtype=dtype)

        # calculate preliminary quantities
        self.method = method
        if method == "central":
            self.scale_r = 0.5 / grid.discretization[0]
        elif method in {"forward", "backward"}:
            self.scale_r = 1 / grid.discretization[0]
        else:
            msg = f"Unknown derivative type `{method}`"
            raise ValueError(msg)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        if self.method == "central":
            r = (data_full[2:] - data_full[:-2]) * self.scale_r
        elif self.method == "forward":
            r = (data_full[2:] - data_full[1:-1]) * self.scale_r
        elif self.method == "backward":
            r = (data_full[1:-1] - data_full[:-2]) * self.scale_r
        # no angular dependence by definition
        return torch.stack((r, torch.zeros_like(r)))


@torch_backend.register_operator(
    PolarSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
class PolarGradientSquared(TorchOperator):
    """Polar gradient squared operator using torch."""

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        central: bool = True,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Polar gradient squared operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            central (bool):
                Whether to use central differences. If `False`, forward and backward
                differences are used.
            dtype:
                The data type of the field
        """
        super().__init__(grid, bcs, dtype=dtype)
        self.central = central
        dr = grid.discretization[0]
        if self.central:
            self.scale = 0.25 / dr**2
        else:
            self.scale = 0.5 / dr**2

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        if self.central:
            # simple squared sum of central differences
            return (data_full[2:] - data_full[:-2]) ** 2 * self.scale  # type: ignore

        term1 = (data_full[2:] - data_full[1:-1]) ** 2
        term2 = (data_full[1:-1] - data_full[:-2]) ** 2
        return (term1 + term2) * self.scale  # type: ignore


@torch_backend.register_operator(PolarSymGrid, "divergence", rank_in=1, rank_out=0)
class PolarDivergence(TorchOperator):
    """Polar divergence operator using torch."""

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Polar divergence operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, bcs, dtype=dtype)

        dr = self.grid.discretization[0]
        self.register_buffer("rs", torch.from_numpy(self.grid.axes_coords[0]))
        self.scale_r = 1 / (2 * dr)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        term1 = (data_full[0, 2:] - data_full[0, :-2]) * self.scale_r
        term2 = data_full[0, 1:-1] / self.rs  # type: ignore
        return term1 + term2  # type: ignore


# @torch_backend.register_operator(
#     PolarSymGrid, "vector_gradient", rank_in=1, rank_out=2
# )
# class PolarVectorGradient(TorchOperator):
#     """Polar vector gradient operator using torch."""

#     rank_in = 1

#     def __init__(
#         self,
#         grid: GridBase,
#         bcs: BoundariesList | None,
#         *,
#         dtype: AnyDType = np.double,
#     ):
#         """Initialize the Polar divergence operator.

#         Args:
#             grid (:class:`~pde.grids.base.GridBase`):
#                 The grid on which the operator acts
#             bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
#                 The boundary conditions applied to the field. If `None`, no boundary
#                 conditions are enforced.
#             dtype:
#                 The data type of the field
#         """
#         super().__init__(grid, bcs, dtype=dtype)

#         dr = self.grid.discretization[0]
#         self.register_buffer("rs", torch.from_numpy(self.grid.axes_coords[0]))
#         self.scale_r = 1 / (2 * dr)

#     def forward(self, arr: Tensor, args=None) -> Tensor:
#         """Fill internal data array, apply operator, and return valid data."""
#         data_full = self.get_full_data(arr, args=args)

#         # assign aliases
#         arr_r, arr_φ = arr
#         out_rr, out_rφ = out[0, 0, :], out[0, 1, :]
#         out_φr, out_φφ = out[1, 0, :], out[1, 1, :]

#         for i in range(1, dim_r + 1):  # iterate radial points
#             out_rr[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
#             out_rφ[i - 1] = -arr_φ[i] / rs[i - 1]
#             out_φr[i - 1] = (arr_φ[i + 1] - arr_φ[i - 1]) * scale_r
#             out_φφ[i - 1] = arr_r[i] / rs[i - 1]

#         term1 = (data_full[0, 2:] - data_full[0, :-2]) * self.scale_r
#         term2 = data_full[0, 1:-1] / self.rs
#         return term1 + term2


__all__ = ["PolarDivergence", "PolarGradient", "PolarGradientSquared", "PolarLaplacian"]
