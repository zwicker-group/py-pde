r"""This module implements differential operators on polar grids.

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_gradient_squared
   make_divergence
   make_vector_gradient
   make_tensor_divergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from ....grids.spherical import PolarSymGrid
from .. import jax_backend

if TYPE_CHECKING:
    import jax

    from ....tools.typing import OperatorImplType


@jax_backend.register_operator(PolarSymGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized laplace operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dr = grid.discretization[0]
    factor_r = 1 / (2 * grid.axes_coords[0] * dr)
    dr_2 = 1 / dr**2

    def laplace(arr: jax.Array) -> jax.Array:
        """Apply Laplace operator to array `arr`"""
        term1 = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) * dr_2
        term2 = (arr[2:] - arr[:-2]) * factor_r
        return term1 + term2  # type: ignore

    return laplace


@jax_backend.register_operator(PolarSymGrid, "gradient", rank_in=0, rank_out=1)
def make_gradient(
    grid: PolarSymGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a discretized gradient operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    if method == "central":
        scale_r = 0.5 / grid.discretization[0]
    elif method in {"forward", "backward"}:
        scale_r = 1 / grid.discretization[0]
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def gradient(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            r = (arr[2:] - arr[:-2]) * scale_r
        elif method == "forward":
            r = (arr[2:] - arr[1:-1]) * scale_r
        elif method == "backward":
            r = (arr[1:-1] - arr[:-2]) * scale_r
        # no angular dependence by definition
        return jnp.stack((r, jnp.zeros_like(r)))

    return gradient


@jax_backend.register_operator(PolarSymGrid, "gradient_squared", rank_in=0, rank_out=0)
def make_gradient_squared(
    grid: PolarSymGrid, *, central: bool = True
) -> OperatorImplType:
    """Make a discretized gradient squared operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dr = grid.discretization[0]

    if central:
        # use central differences
        scale = 0.25 / dr**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            return (arr[2:] - arr[:-2]) ** 2 * scale  # type: ignore

    else:
        # use forward and backward differences
        scale = 0.5 / dr**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            return ((arr[2:] - arr[1:-1]) ** 2 + (arr[1:-1] - arr[:-2]) ** 2) * scale  # type: ignore

    return gradient_squared


@jax_backend.register_operator(PolarSymGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized divergence operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    scale_r = 1 / (2 * dr)

    def divergence(arr: jax.Array) -> jax.Array:
        """Apply divergence operator to array `arr`"""
        return (arr[0, 2:] - arr[0, :-2]) * scale_r + arr[0, 1:-1] / rs  # type: ignore

    return divergence


@jax_backend.register_operator(PolarSymGrid, "vector_gradient", rank_in=1, rank_out=2)
def make_vector_gradient(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized vector gradient operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    def vector_gradient(arr: jax.Array) -> jax.Array:
        """Apply vector gradient operator to array `arr`"""
        arr_r, arr_φ = arr[0], arr[1]

        out_rr = (arr_r[2:] - arr_r[:-2]) * scale_r
        out_rφ = -arr_φ[1:-1] / rs
        out_φr = (arr_φ[2:] - arr_φ[:-2]) * scale_r
        out_φφ = arr_r[1:-1] / rs

        return jnp.stack([jnp.stack([out_rr, out_rφ]), jnp.stack([out_φr, out_φφ])])

    return vector_gradient


@jax_backend.register_operator(PolarSymGrid, "tensor_divergence", rank_in=2, rank_out=1)
def make_tensor_divergence(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized tensor divergence operator for a polar grid.

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    def tensor_divergence(arr: jax.Array) -> jax.Array:
        """Apply tensor divergence operator to array `arr`"""
        arr_rr, arr_rφ = arr[0, 0], arr[0, 1]
        arr_φr, arr_φφ = arr[1, 0], arr[1, 1]

        out_r = (arr_rr[2:] - arr_rr[:-2]) * scale_r + (
            arr_rr[1:-1] - arr_φφ[1:-1]
        ) / rs
        out_φ = (arr_φr[2:] - arr_φr[:-2]) * scale_r + (
            arr_rφ[1:-1] + arr_φr[1:-1]
        ) / rs

        return jnp.stack((out_r, out_φ))

    return tensor_divergence


__all__ = [
    "make_divergence",
    "make_gradient",
    "make_gradient_squared",
    "make_laplace",
    "make_tensor_divergence",
    "make_vector_gradient",
]
