r"""This module implements differential operators on polar grids.

.. autosummary::
   :nosignatures:

   PolarLaplacian
   PolarGradient
   PolarGradientSquared
   PolarDivergence
   PolarVectorGradient
   PolarTensorDivergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ....grids import GridBase, PolarSymGrid
from ....tools.docstrings import fill_in_docstring
from ..backend import TorchBackend
from .common import TorchDifferentialOperator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from ....grids.boundaries import BoundariesList


@TorchBackend.register_operator(PolarSymGrid, "laplace", rank_in=0, rank_out=0)
@fill_in_docstring
class PolarLaplacian(TorchDifferentialOperator):
    """Polar Laplace using torch.

    {DESCR_POLAR_GRID}
    """

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
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
        self.register_array("factor_r", factor_r)
        self.dr_2 = 1 / dr**2

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        term1 = (data_full[2:] - 2 * data_full[1:-1] + data_full[:-2]) * self.dr_2
        term2 = (data_full[2:] - data_full[:-2]) * self.factor_r  # type: ignore
        return term1 + term2  # type: ignore


@TorchBackend.register_operator(PolarSymGrid, "gradient", rank_in=0, rank_out=1)
@fill_in_docstring
class PolarGradient(TorchDifferentialOperator):
    """Polar gradient operator using torch.

    {DESCR_POLAR_GRID}
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


@TorchBackend.register_operator(PolarSymGrid, "gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
class PolarGradientSquared(TorchDifferentialOperator):
    """Polar gradient squared operator using torch.

    {DESCR_POLAR_GRID}
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


@TorchBackend.register_operator(PolarSymGrid, "divergence", rank_in=1, rank_out=0)
@fill_in_docstring
class PolarDivergence(TorchDifferentialOperator):
    """Polar divergence operator using torch.

    {DESCR_POLAR_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
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
        self.register_array("rs", self.grid.axes_coords[0])
        self.scale_r = 1 / (2 * dr)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        term1 = (data_full[0, 2:] - data_full[0, :-2]) * self.scale_r
        term2 = data_full[0, 1:-1] / self.rs  # type: ignore
        return term1 + term2  # type: ignore


@TorchBackend.register_operator(PolarSymGrid, "vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
class PolarVectorGradient(TorchDifferentialOperator):
    """Polar vector gradient operator using torch.

    {DESCR_POLAR_GRID}
    """

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the Polar vector gradient operator.

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
        self.register_array("rs", self.grid.axes_coords[0])
        self.scale_r = 1 / (2 * dr)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        arr_r, arr_φ = data_full[0], data_full[1]

        out_rr = (arr_r[2:] - arr_r[:-2]) * self.scale_r
        out_rφ = -arr_φ[1:-1] / self.rs  # type: ignore
        out_φr = (arr_φ[2:] - arr_φ[:-2]) * self.scale_r
        out_φφ = arr_r[1:-1] / self.rs  # type: ignore

        return torch.stack(
            [torch.stack([out_rr, out_rφ]), torch.stack([out_φr, out_φφ])]
        )


@TorchBackend.register_operator(
    PolarSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
@fill_in_docstring
class PolarTensorDivergence(TorchDifferentialOperator):
    """Polar tensor divergence operator using torch.

    {DESCR_POLAR_GRID}
    """

    rank_in = 2

    def __init__(
        self,
        grid: GridBase,
        bcs: BoundariesList | None,
        *,
        dtype: np.dtype,
    ):
        """Initialize the Polar tensor divergence operator.

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
        self.register_array("rs", self.grid.axes_coords[0])
        self.scale_r = 1 / (2 * dr)

    def forward(self, arr: Tensor, args=None) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr, args=args)

        arr_rr, arr_rφ = data_full[0, 0], data_full[0, 1]
        arr_φr, arr_φφ = data_full[1, 0], data_full[1, 1]

        out_r = (arr_rr[2:] - arr_rr[:-2]) * self.scale_r
        out_r += (arr_rr[1:-1] - arr_φφ[1:-1]) / self.rs  # type: ignore
        out_φ = (arr_φr[2:] - arr_φr[:-2]) * self.scale_r
        out_φ += (arr_rφ[1:-1] + arr_φr[1:-1]) / self.rs  # type: ignore

        return torch.stack((out_r, out_φ))


__all__ = [
    "PolarDivergence",
    "PolarGradient",
    "PolarGradientSquared",
    "PolarLaplacian",
    "PolarTensorDivergence",
    "PolarVectorGradient",
]
