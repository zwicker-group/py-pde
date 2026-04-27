r"""This module implements differential operators on spherical grids.

.. autosummary::
   :nosignatures:

   CylindricalLaplacian
   CylindricalGradient
   CylindricalGradientSquared
   CylindricalDivergence
   CylindricalVectorGradient
   CylindricalVectorLaplacian
   CylindricalTensorDivergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....grids import CylindricalSymGrid, GridBase
from ....tools.docstrings import fill_in_docstring
from ..backend import TorchBackend
from .common import TorchDifferentialOperator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from ....grids.boundaries import BoundariesList


@TorchBackend.register_operator(CylindricalSymGrid, "laplace", rank_in=0, rank_out=0)
@fill_in_docstring
class CylindricalLaplacian(TorchDifferentialOperator):
    """Cylindrical Laplace using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 0

    def __init__(self, grid: GridBase, bcs: BoundariesList | None, *, dtype: np.dtype):
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
        self.register_array("factor_r", factor_r[:, None])

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


@TorchBackend.register_operator(CylindricalSymGrid, "gradient", rank_in=0, rank_out=1)
@fill_in_docstring
class CylindricalGradient(TorchDifferentialOperator):
    """Cylindrical gradient operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
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
        self.result_shape = (3, *grid.shape)
        self.scale_r, self.scale_z = 1 / (2 * grid.discretization)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        result = torch.zeros(self.result_shape, dtype=arr.dtype, device=arr.device)

        result[0] = (data_full[2:, 1:-1] - data_full[:-2, 1:-1]) * self.scale_r  # r
        result[1] = (data_full[1:-1, 2:] - data_full[1:-1, :-2]) * self.scale_z  # z
        # phi = torch.zeros_like(r)
        return result


@TorchBackend.register_operator(
    CylindricalSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
@fill_in_docstring
class CylindricalGradientSquared(TorchDifferentialOperator):
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
        dtype: np.dtype,
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


@TorchBackend.register_operator(CylindricalSymGrid, "divergence", rank_in=1, rank_out=0)
@fill_in_docstring
class CylindricalDivergence(TorchDifferentialOperator):
    """Cylindrical divergence operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 1

    def __init__(self, grid: GridBase, bcs: BoundariesList | None, *, dtype: np.dtype):
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
        self.register_array("rs", rs[:, None])

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        arr_r, arr_z = data_full[0], data_full[1]
        return (  # type: ignore
            arr_r[1:-1, 1:-1] / self.rs  # type: ignore
            + (arr_r[2:, 1:-1] - arr_r[:-2, 1:-1]) * self.scale_r
            + (arr_z[1:-1, 2:] - arr_z[1:-1, :-2]) * self.scale_z
        )


@TorchBackend.register_operator(
    CylindricalSymGrid, "vector_gradient", rank_in=1, rank_out=2
)
@fill_in_docstring
class CylindricalVectorGradient(TorchDifferentialOperator):
    """Cylindrical vector gradient operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the Cylindrical vector gradient operator.

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

        self.scale_r, self.scale_z = 0.5 / grid.discretization
        self.result_shape = (3, 3, *grid.shape)
        rs = grid.axes_coords[0]
        self.register_array("rs", rs[:, None])

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        result = torch.zeros(self.result_shape, dtype=arr.dtype, device=arr.device)

        arr_r, arr_z, arr_φ = data_full[0], data_full[1], data_full[2]

        # radial derivatives
        result[0, 0] = (arr_r[2:, 1:-1] - arr_r[:-2, 1:-1]) * self.scale_r  # rr
        result[1, 0] = (arr_z[2:, 1:-1] - arr_z[:-2, 1:-1]) * self.scale_r  # zr
        result[2, 0] = (arr_φ[2:, 1:-1] - arr_φ[:-2, 1:-1]) * self.scale_r  # φr

        # phi-curvature terms
        result[0, 2] = -arr_φ[1:-1, 1:-1] / self.rs  # type: ignore  # rφ
        result[2, 2] = arr_r[1:-1, 1:-1] / self.rs  # type: ignore  # φφ
        # out_zφ = torch.zeros_like(out_rr)

        # axial derivatives
        result[0, 1] = (arr_r[1:-1, 2:] - arr_r[1:-1, :-2]) * self.scale_z  # rz
        result[2, 1] = (arr_φ[1:-1, 2:] - arr_φ[1:-1, :-2]) * self.scale_z  # φz
        result[1, 1] = (arr_z[1:-1, 2:] - arr_z[1:-1, :-2]) * self.scale_z  # zz
        return result


@TorchBackend.register_operator(
    CylindricalSymGrid, "vector_laplace", rank_in=1, rank_out=1
)
@fill_in_docstring
class CylindricalVectorLaplacian(TorchDifferentialOperator):
    """Cylindrical vector Laplacian operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the Cylindrical vector Laplacian operator.

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

        rs = grid.axes_coords[0]
        self.result_shape = (3, *grid.shape)
        self.register_array("rs", rs[:, None])
        dr, dz = grid.discretization
        self.s1 = 1 / (2 * dr)
        self.s2 = 1 / dr**2
        self.scale_z = 1 / dz**2

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        result = torch.empty(self.result_shape, dtype=arr.dtype, device=arr.device)

        arr_r, arr_z, arr_φ = data_full[0], data_full[1], data_full[2]

        f_r_l = arr_r[:-2, 1:-1]
        f_r_m = arr_r[1:-1, 1:-1]
        f_r_h = arr_r[2:, 1:-1]
        result[0] = (  # r component
            (arr_r[1:-1, 2:] - 2 * f_r_m + arr_r[1:-1, :-2]) * self.scale_z
            - f_r_m / self.rs**2  # type: ignore
            + (f_r_h - f_r_l) * self.s1 / self.rs
            + (f_r_h - 2 * f_r_m + f_r_l) * self.s2
        )

        f_φ_l = arr_φ[:-2, 1:-1]
        f_φ_m = arr_φ[1:-1, 1:-1]
        f_φ_h = arr_φ[2:, 1:-1]
        result[2] = (  # φ component
            (arr_φ[1:-1, 2:] - 2 * f_φ_m + arr_φ[1:-1, :-2]) * self.scale_z
            - f_φ_m / self.rs**2  # type: ignore
            + (f_φ_h - f_φ_l) * self.s1 / self.rs
            + (f_φ_h - 2 * f_φ_m + f_φ_l) * self.s2
        )

        f_z_l = arr_z[:-2, 1:-1]
        f_z_m = arr_z[1:-1, 1:-1]
        f_z_h = arr_z[2:, 1:-1]
        result[1] = (  # z component
            (arr_z[1:-1, 2:] - 2 * f_z_m + arr_z[1:-1, :-2]) * self.scale_z
            + (f_z_h - f_z_l) * self.s1 / self.rs
            + (f_z_h - 2 * f_z_m + f_z_l) * self.s2
        )
        return result


@TorchBackend.register_operator(
    CylindricalSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
@fill_in_docstring
class CylindricalTensorDivergence(TorchDifferentialOperator):
    """Cylindrical tensor divergence operator using torch.

    {DESCR_CYLINDRICAL_GRID}
    """

    rank_in = 2

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the Cylindrical tensor divergence operator.

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

        rs = grid.axes_coords[0]
        self.result_shape = (3, *grid.shape)
        self.register_array("rs", rs[:, None])
        self.scale_r, self.scale_z = 0.5 / grid.discretization

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        result = torch.empty(self.result_shape, dtype=arr.dtype, device=arr.device)

        arr_rr, arr_rz, arr_rφ = data_full[0, 0], data_full[0, 1], data_full[0, 2]
        arr_zr, arr_zz = data_full[1, 0], data_full[1, 1]
        arr_φr, arr_φz, arr_φφ = data_full[2, 0], data_full[2, 1], data_full[2, 2]

        result[0] = (  # r component
            (arr_rz[1:-1, 2:] - arr_rz[1:-1, :-2]) * self.scale_z
            + (arr_rr[2:, 1:-1] - arr_rr[:-2, 1:-1]) * self.scale_r
            + (arr_rr[1:-1, 1:-1] - arr_φφ[1:-1, 1:-1]) / self.rs  # type: ignore
        )

        result[2] = (  # φ component
            (arr_φz[1:-1, 2:] - arr_φz[1:-1, :-2]) * self.scale_z
            + (arr_φr[2:, 1:-1] - arr_φr[:-2, 1:-1]) * self.scale_r
            + (arr_rφ[1:-1, 1:-1] + arr_φr[1:-1, 1:-1]) / self.rs  # type: ignore
        )

        result[1] = (  # z component
            (arr_zz[1:-1, 2:] - arr_zz[1:-1, :-2]) * self.scale_z
            + (arr_zr[2:, 1:-1] - arr_zr[:-2, 1:-1]) * self.scale_r
            + arr_zr[1:-1, 1:-1] / self.rs  # type: ignore
        )
        return result


__all__ = [
    "CylindricalDivergence",
    "CylindricalGradient",
    "CylindricalGradientSquared",
    "CylindricalLaplacian",
    "CylindricalTensorDivergence",
    "CylindricalVectorGradient",
    "CylindricalVectorLaplacian",
]
