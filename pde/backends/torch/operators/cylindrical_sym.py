r"""This module implements differential operators on spherical grids.

.. autosummary::
   :nosignatures:

   CylindricalLaplacian
   CylindricalGradient
   CylindricalGradientSquared
   CylindricalDivergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ....grids import CylindricalSymGrid, GridBase
from ....tools.docstrings import fill_in_docstring
from .. import torch_backend
from .common import TorchOperator

if TYPE_CHECKING:
    from torch import Tensor

    from ....grids.boundaries import BoundariesList
    from ..utils import AnyDType


@torch_backend.register_operator(CylindricalSymGrid, "laplace", rank_in=0, rank_out=0)
@fill_in_docstring
class CylindricalLaplacian(TorchOperator):
    """Cylindrical Laplace using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self, grid: GridBase, bcs: BoundariesList | None, *, dtype: AnyDType = np.double
    ):
        """Initialize the Cylindrical Laplacian operator.

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
        self.dr_2, self.dz_2 = 1 / grid.discretization**2
        factor_r = 1 / (2 * grid.axes_coords[0] * dr)
        self.register_buffer("factor_r", torch.from_numpy(factor_r[:, None]))

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        arr_z_l, arr_z_h = data_full[1:-1, :-2], data_full[1:-1, 2:]
        arr_mid = data_full[1:-1, 1:-1]
        arr_r_l, arr_r_h = data_full[:-2, 1:-1], data_full[2:, 1:-1]
        return (  # type: ignore
            (arr_r_h - 2 * arr_mid + arr_r_l) * self.dr_2
            + (arr_r_h - arr_r_l) * self.factor_r  # type: ignore
            + (arr_z_l - 2 * arr_mid + arr_z_h) * self.dz_2
        )


@torch_backend.register_operator(CylindricalSymGrid, "gradient", rank_in=0, rank_out=1)
@fill_in_docstring
class CylindricalGradient(TorchOperator):
    """Cylindrical gradient operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cylindrical gradient operator.

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
        self.scale_r, self.scale_z = 1 / (2 * grid.discretization)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        r = (data_full[2:, 1:-1] - data_full[:-2, 1:-1]) * self.scale_r
        z = (data_full[1:-1, 2:] - data_full[1:-1, :-2]) * self.scale_z
        phi = torch.zeros_like(r)
        return torch.stack((r, z, phi))


@torch_backend.register_operator(
    CylindricalSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
@fill_in_docstring
class CylindricalGradientSquared(TorchOperator):
    """Cylindrical gradient squared operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        central: bool = True,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cylindrical gradient squared operator.

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
        if self.central:
            self.scale_r, self.scale_z = 0.25 / grid.discretization**2
        else:
            self.scale_r, self.scale_z = 0.5 / grid.discretization**2

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        if self.central:
            # simple squared sum of central differences
            r = (data_full[2:, 1:-1] - data_full[:-2, 1:-1]) ** 2 * self.scale_r
            z = (data_full[1:-1, 2:] - data_full[1:-1, :-2]) ** 2 * self.scale_z
            return r + z  # type: ignore

        arr_z_h = data_full[1:-1, 2:]
        arr_c = data_full[1:-1, 1:-1]
        arr_z_l = data_full[1:-1, :-2]
        term_r = (arr[2:, 1:-1] - arr_c) ** 2 + (arr_c - arr[:-2, 1:-1]) ** 2
        term_z = (arr_z_h - arr_c) ** 2 + (arr_c - arr_z_l) ** 2
        return term_r * self.scale_r + term_z * self.scale_z  # type: ignore


@torch_backend.register_operator(
    CylindricalSymGrid, "divergence", rank_in=1, rank_out=0
)
@fill_in_docstring
class CylindricalDivergence(TorchOperator):
    """Cylindrical divergence operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 1

    def __init__(
        self, grid: GridBase, bcs: BoundariesList | None, *, dtype: AnyDType = np.double
    ):
        """Initialize the Cylindrical divergence operator.

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

        self.scale_r, self.scale_z = 1 / (2 * grid.discretization)
        rs = grid.axes_coords[0]
        self.register_buffer("rs", torch.from_numpy(rs[:, None]))

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        arr_r, arr_z = data_full[0], data_full[1]
        return (  # type: ignore
            arr_r[1:-1, 1:-1] / self.rs  # type: ignore
            + (arr_r[2:, 1:-1] - arr_r[:-2, 1:-1]) * self.scale_r
            + (arr_z[1:-1, 2:] - arr_z[1:-1, :-2]) * self.scale_z
        )


# @torch_backend.register_operator(
#     CylindricalSymGrid, "vector_gradient", rank_in=1, rank_out=2
# )
# class CylindricalVectorGradient(TorchOperator):
#     """Cylindrical vector gradient operator using torch."""

#     rank_in = 1

#     def __init__(
#         self,
#         grid: GridBase,
#         bcs: BoundariesList | None,
#         *,
#         dtype: AnyDType = np.double,
#     ):
#         """Initialize the Cylindrical divergence operator.

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


__all__ = [
    "CylindricalDivergence",
    "CylindricalGradient",
    "CylindricalGradientSquared",
    "CylindricalLaplacian",
]
