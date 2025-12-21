"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   CartesianLaplacian

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

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

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian Laplacian operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.scale = self.grid.discretization**-2

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr)

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


@torch_backend.register_operator(CartesianGrid, "gradient", rank_in=0, rank_out=1)
class CartesianGradient(TorchOperator):
    """Cartesian gradient operator using torch."""

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian gradient operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.scale = 0.5 / self.grid.discretization

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr)

        if self.grid.num_axes == 1:
            # one-dimensional grids support various implementations of finite difference
            x = (data_full[2:] - data_full[:-2]) * self.scale[0]
            return x[None, :]  # type: ignore

        if self.grid.num_axes == 2:
            # two-dimensional grids only support central differences
            x = (data_full[2:, 1:-1] - data_full[:-2, 1:-1]) * self.scale[0]
            y = (data_full[1:-1, 2:] - data_full[1:-1, :-2]) * self.scale[1]
            return torch.stack((x, y))

        if self.grid.num_axes == 3:
            # three-dimensional grids only support central differences
            x = (data_full[2:, 1:-1, 1:-1] - data_full[:-2, 1:-1, 1:-1]) * self.scale[0]
            y = (data_full[1:-1, 2:, 1:-1] - data_full[1:-1, :-2, 1:-1]) * self.scale[1]
            z = (data_full[1:-1, 1:-1, 2:] - data_full[1:-1, 1:-1, :-2]) * self.scale[2]
            return torch.stack((x, y, z))

        raise NotImplementedError


@torch_backend.register_operator(
    CartesianGrid, "gradient_squared", rank_in=0, rank_out=0
)
class CartesianGradientSquared(TorchOperator):
    """Cartesian gradient squared operator using torch."""

    rank_in = 0

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        central: bool = True,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian gradient squared operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            central (bool):
                Whether to use central differences. If `False`, forward and backward
                differences are used.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.central = central
        if self.central:
            self.grad_central = CartesianGradient(grid, boundaries, dtype=dtype)
        else:
            self.scale = 0.5 / self.grid.discretization**2

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        if self.central:
            # simple squared sum of central differences
            return torch.sum(self.grad_central(arr) ** 2, dim=0)

        # use forward and backward differences
        data_full = self.get_full_data(arr)
        if self.grid.num_axes == 1:
            # use forward and backward differences
            diff_l = (data_full[2:] - data_full[1:-1]) ** 2
            diff_r = (data_full[1:-1] - data_full[:-2]) ** 2
            return (diff_l + diff_r) * self.scale[0]  # type: ignore

        if self.grid.num_axes == 2:
            # two-dimensional grids only support central differences
            x = (
                (data_full[2:, 1:-1] - data_full[1:-1, 1:-1]) ** 2
                + (data_full[1:-1, 1:-1] - data_full[:-2, 1:-1]) ** 2
            ) * self.scale[0]
            y = (
                (data_full[1:-1, 2:] - data_full[1:-1, 1:-1]) ** 2
                + (data_full[1:-1, 1:-1] - data_full[1:-1, :-2]) ** 2
            ) * self.scale[1]
            return x + y  # type: ignore

        if self.grid.num_axes == 3:
            # three-dimensional grids only support central differences
            x = (
                (data_full[2:, 1:-1, 1:-1] - data_full[1:-1, 1:-1, 1:-1]) ** 2
                + (data_full[1:-1, 1:-1, 1:-1] - data_full[:-2, 1:-1, 1:-1]) ** 2
            ) * self.scale[0]
            y = (
                (data_full[1:-1, 2:, 1:-1] - data_full[1:-1, 1:-1, 1:-1]) ** 2
                + (data_full[1:-1, 1:-1, 1:-1] - data_full[1:-1, :-2, 1:-1]) ** 2
            ) * self.scale[1]
            z = (
                (data_full[1:-1, 1:-1, 2:] - data_full[1:-1, 1:-1, 1:-1]) ** 2
                + (data_full[1:-1, 1:-1, 1:-1] - data_full[1:-1, 1:-1, :-2]) ** 2
            ) * self.scale[2]
            return x + y + z  # type: ignore

        raise NotImplementedError


@torch_backend.register_operator(CartesianGrid, "divergence", rank_in=1, rank_out=0)
class CartesianDivergence(TorchOperator):
    """Cartesian divergence operator using torch."""

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian divergence operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.scale = 0.5 / self.grid.discretization

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        data_full = self.get_full_data(arr)

        if self.grid.num_axes == 1:
            return (data_full[0, 2:] - data_full[0, :-2]) * self.scale[0]  # type: ignore

        if self.grid.num_axes == 2:
            # two-dimensional grids only support central differences
            x = (data_full[0, 2:, 1:-1] - data_full[0, :-2, 1:-1]) * self.scale[0]
            y = (data_full[1, 1:-1, 2:] - data_full[1, 1:-1, :-2]) * self.scale[1]
            return x + y  # type: ignore

        if self.grid.num_axes == 3:
            # three-dimensional grids only support central differences
            x = data_full[0, 2:, 1:-1, 1:-1] - data_full[0, :-2, 1:-1, 1:-1]
            y = data_full[1, 1:-1, 2:, 1:-1] - data_full[1, 1:-1, :-2, 1:-1]
            z = data_full[2, 1:-1, 1:-1, 2:] - data_full[2, 1:-1, 1:-1, :-2]
            return x * self.scale[0] + y * self.scale[1] + z * self.scale[2]  # type: ignore

        raise NotImplementedError


@torch_backend.register_operator(
    CartesianGrid, "vector_gradient", rank_in=1, rank_out=2
)
class CartesianVectorGradient(TorchOperator):
    """Cartesian vector gradient operator using torch."""

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian vector gradient operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.grad = CartesianGradient(grid, boundaries, dtype=dtype)

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        return torch.stack(tuple(self.grad(arr[i]) for i in range(self.grid.num_axes)))


@torch_backend.register_operator(CartesianGrid, "vector_laplace", rank_in=1, rank_out=1)
class CartesianVectorLaplacian(TorchOperator):
    """Cartesian vector Laplacian operator using torch."""

    rank_in = 1

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian vector Laplacian operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.lap = CartesianLaplacian(grid, boundaries, dtype=dtype)

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        return torch.stack(tuple(self.lap(arr[i]) for i in range(self.grid.num_axes)))


@torch_backend.register_operator(
    CartesianGrid, "tensor_divergence", rank_in=2, rank_out=1
)
class CartesianTensorDivergence(TorchOperator):
    """Cartesian tensor divergence operator using torch."""

    rank_in = 2

    def __init__(
        self,
        grid: GridBase,
        boundaries: BoundariesList | None,
        *,
        dtype: AnyDType = np.double,
    ):
        """Initialize the Cartesian tensor divergence operator.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the operator acts
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesList` or None):
                The boundary conditions applied to the field. If `None`, no boundary
                conditions are enforced.
            dtype:
                The data type of the field
        """
        super().__init__(grid, boundaries, dtype=dtype)
        self.div = CartesianDivergence(grid, boundaries, dtype=dtype)

    def forward(self, arr: Tensor) -> Tensor:
        """Fill internal data array, apply operator, and return valid data."""
        return torch.stack(tuple(self.div(arr[i]) for i in range(self.grid.num_axes)))


__all__ = [
    "CartesianDivergence",
    "CartesianGradient",
    "CartesianGradientSquared",
    "CartesianLaplacian",
    "CartesianTensorDivergence",
    "CartesianVectorGradient",
    "CartesianVectorLaplacian",
]
