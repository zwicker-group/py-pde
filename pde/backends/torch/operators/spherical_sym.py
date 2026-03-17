r"""This module implements differential operators on spherical grids.

.. autosummary::
   :nosignatures:

   SphericalLaplacian
   SphericalGradient
   SphericalGradientSquared
   SphericalDivergence
   SphericalVectorGradient
   SphericalTensorDivergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .... import config
from ....grids import GridBase, SphericalSymGrid
from ....tools.docstrings import fill_in_docstring
from .. import torch_backend
from .common import TorchDifferentialOperator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from ....grids.boundaries import BoundariesList


@torch_backend.register_operator(SphericalSymGrid, "laplace", rank_in=0, rank_out=0)
@fill_in_docstring
class SphericalLaplacian(TorchDifferentialOperator):
    """Spherical Laplace using torch.

    {DESCR_SPHERICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
        conservative: bool | None = None,
    ):
        """Initialize the Spherical Laplacian operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
            conservative (bool):
                Flag indicating whether the Laplace operator should be conservative
                (which results in slightly slower computations). Conservative operators
                ensure mass conservation. If `None`, the value is read from the
                configuration option `operators.conservative_stencil`.
        """
        super().__init__(grid, bcs, dtype=dtype)

        if conservative is None:
            conservative = config["operators.conservative_stencil"]
        self.conservative = conservative

        # calculate preliminary quantities
        dr = grid.discretization[0]
        self.dr = dr
        rs = grid.axes_coords[0]
        self.dr_2 = 1 / dr**2
        if self.conservative:
            # create a conservative spherical laplace operator
            rl = rs - dr / 2  # inner radii of spherical shells
            rh = rs + dr / 2  # outer radii
            volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
            self.register_array("factor_l", rl**2 / (dr * volumes))
            self.register_array("factor_h", rh**2 / (dr * volumes))
        else:
            self.register_array("factor", 1 / (rs * dr))

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        if self.conservative:
            term_h = self.factor_h * (data_full[2:] - data_full[1:-1])  # type: ignore
            term_l = self.factor_l * (data_full[1:-1] - data_full[:-2])  # type: ignore
            return term_h - term_l

        term1 = (data_full[2:] - 2 * data_full[1:-1] + data_full[:-2]) * self.dr_2
        term2 = self.factor * (data_full[2:] - data_full[:-2])  # type: ignore
        return term1 + term2  # type: ignore


@torch_backend.register_operator(SphericalSymGrid, "gradient", rank_in=0, rank_out=1)
@fill_in_docstring
class SphericalGradient(TorchDifferentialOperator):
    """Spherical gradient operator using torch.

    {DESCR_SPHERICAL_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        """Initialize the Spherical gradient operator.

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
        return torch.stack((r, torch.zeros_like(r), torch.zeros_like(r)))


@torch_backend.register_operator(
    SphericalSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
@fill_in_docstring
class SphericalGradientSquared(TorchDifferentialOperator):
    """Spherical gradient squared operator using torch.

    {DESCR_SPHERICAL_GRID}
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
        """Initialize the Spherical gradient squared operator.

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


@torch_backend.register_operator(SphericalSymGrid, "divergence", rank_in=1, rank_out=0)
@fill_in_docstring
class SphericalDivergence(TorchDifferentialOperator):
    """Spherical divergence operator using torch.

    {DESCR_SPHERICAL_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
        conservative: bool | None = None,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        """Initialize the Spherical divergence operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
            conservative (bool):
                Flag indicating whether the operator should be conservative (which
                results in slightly slower computations). Conservative operators ensure
                mass conservation. If `None`, the value is read from the configuration
                option `operators.conservative_stencil`.
            method (str):
                The method for calculating the derivative. Possible values are
                'central', 'forward', and 'backward'.
        """
        super().__init__(grid, bcs, dtype=dtype)

        if conservative is None:
            conservative = config["operators.conservative_stencil"]
        self.conservative = conservative
        self.method = method

        dr = self.grid.discretization[0]
        self.dr = dr
        rs = self.grid.axes_coords[0]
        self.register_array("rs", rs)
        self.scale_r = 1 / (2 * dr)

        # create a conservative spherical divergence operator
        if self.conservative:
            rl = rs - dr / 2  # inner radii of spherical shells
            rh = rs + dr / 2  # outer radii
            volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
            self.register_array("factor_l", rl**2 / (2 * volumes))
            self.register_array("factor_h", rh**2 / (2 * volumes))
        else:
            self.register_array("factor", 1 / (rs * dr))

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)
        arr_r = data_full[0]

        if self.conservative:
            if self.method == "central":
                term_h = self.factor_h * (arr_r[1:-1] + arr_r[2:])  # type: ignore
                term_l = self.factor_l * (arr_r[:-2] + arr_r[1:-1])  # type: ignore
            elif self.method == "forward":
                term_h = 2 * self.factor_h * arr_r[2:]  # type: ignore
                term_l = 2 * self.factor_l * arr_r[1:-1]  # type: ignore
            elif self.method == "backward":
                term_h = 2 * self.factor_h * arr_r[1:-1]  # type: ignore
                term_l = 2 * self.factor_l * arr_r[:-2]  # type: ignore
            return term_h - term_l

        # non-conservative implementation
        if self.method == "central":
            diff_r = (arr_r[2:] - arr_r[:-2]) / (2 * self.dr)
        elif self.method == "forward":
            diff_r = (arr_r[2:] - arr_r[1:-1]) / self.dr
        elif self.method == "backward":
            diff_r = (arr_r[1:-1] - arr_r[:-2]) / self.dr
        return diff_r + self.factor * arr_r[1:-1]  # type: ignore


@torch_backend.register_operator(
    SphericalSymGrid, "vector_gradient", rank_in=1, rank_out=2
)
@fill_in_docstring
class SphericalVectorGradient(TorchDifferentialOperator):
    """Spherical vector gradient operator using torch.

    {DESCR_SPHERICAL_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        """Initialize the Spherical vector gradient operator.

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

        self.method = method
        rs = grid.axes_coords[0]
        self.register_array("rs", rs)
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

        arr_r = data_full[0]

        if self.method == "central":
            out_rr = (arr_r[2:] - arr_r[:-2]) * self.scale_r
        elif self.method == "forward":
            out_rr = (arr_r[2:] - arr_r[1:-1]) * self.scale_r
        elif self.method == "backward":
            out_rr = (arr_r[1:-1] - arr_r[:-2]) * self.scale_r

        out_diag = arr_r[1:-1] / self.rs  # type: ignore
        zeros = torch.zeros_like(out_rr)

        return torch.stack(
            [
                torch.stack([out_rr, zeros, zeros]),
                torch.stack([zeros, out_diag, zeros]),
                torch.stack([zeros, zeros, out_diag]),
            ]
        )


@torch_backend.register_operator(
    SphericalSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
@fill_in_docstring
class SphericalTensorDivergence(TorchDifferentialOperator):
    """Spherical tensor divergence operator using torch.

    {DESCR_SPHERICAL_GRID}
    """

    rank_in = 2

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
        conservative: bool | None = False,
    ):
        """Initialize the Spherical tensor divergence operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
            conservative (bool):
                Flag indicating whether the operator should be conservative (which
                results in slightly slower computations). Conservative operators ensure
                mass conservation. If `None`, the value is read from the configuration
                option `operators.conservative_stencil`.
        """
        super().__init__(grid, bcs, dtype=dtype)

        if conservative is None:
            conservative = config["operators.conservative_stencil"]
        self.conservative = conservative

        rs = grid.axes_coords[0]
        dr = grid.discretization[0]
        self.register_array("rs", rs)
        self.scale_r = 1 / (2 * dr)

        if self.conservative:
            rl = rs - dr / 2  # inner radii
            rh = rs + dr / 2  # outer radii
            volumes = (rh**3 - rl**3) / 3
            self.register_array("factor_l", rl**2 / (2 * volumes))
            self.register_array("factor_h", rh**2 / (2 * volumes))
            self.register_array("area_factor", (rh**2 - rl**2) / volumes)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        if self.conservative:
            arr_rr = data_full[0, 0]
            arr_φφ = data_full[2, 2]

            term_r_h = self.factor_h * (arr_rr[1:-1] + arr_rr[2:])  # type: ignore
            term_r_l = self.factor_l * (arr_rr[:-2] + arr_rr[1:-1])  # type: ignore
            out_r = term_r_h - term_r_l - self.area_factor * arr_φφ[1:-1]  # type: ignore
            zeros = torch.zeros_like(out_r)
            return torch.stack([out_r, zeros, zeros])

        # non-conservative implementation
        arr_rr = data_full[0, 0]
        arr_rφ = data_full[0, 2]
        arr_θr = data_full[1, 0]
        arr_φr = data_full[2, 0]
        arr_φφ = data_full[2, 2]

        out_r = (arr_rr[2:] - arr_rr[:-2]) * self.scale_r
        out_r += 2 * (arr_rr[1:-1] - arr_φφ[1:-1]) / self.rs  # type: ignore
        out_θ = (arr_θr[2:] - arr_θr[:-2]) * self.scale_r + 2 * arr_θr[1:-1] / self.rs  # type: ignore
        out_φ = (arr_φr[2:] - arr_φr[:-2]) * self.scale_r
        out_φ += (2 * arr_φr[1:-1] + arr_rφ[1:-1]) / self.rs  # type: ignore

        return torch.stack([out_r, out_θ, out_φ])


__all__ = [
    "SphericalDivergence",
    "SphericalGradient",
    "SphericalGradientSquared",
    "SphericalLaplacian",
    "SphericalTensorDivergence",
    "SphericalVectorGradient",
]
