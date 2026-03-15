r"""This module implements differential operators on cylindrical grids.

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_gradient_squared
   make_divergence
   make_vector_gradient
   make_vector_laplace
   make_tensor_divergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from ....grids.cylindrical import CylindricalSymGrid
from .. import jax_backend

if TYPE_CHECKING:
    import jax

    from ....tools.typing import OperatorImplType


@jax_backend.register_operator(CylindricalSymGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized laplace operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dr = grid.discretization[0]
    dr_2, dz_2 = 1 / grid.discretization**2
    factor_r = 1 / (2 * grid.axes_coords[0] * dr)

    def laplace(arr: jax.Array) -> jax.Array:
        """Apply Laplace operator to array `arr`"""
        arr_mid = arr[1:-1, 1:-1]
        arr_r_l, arr_r_h = arr[:-2, 1:-1], arr[2:, 1:-1]
        arr_z_l, arr_z_h = arr[1:-1, :-2], arr[1:-1, 2:]
        return (  # type: ignore
            (arr_r_h - 2 * arr_mid + arr_r_l) * dr_2
            + (arr_r_h - arr_r_l) * factor_r[:, None]
            + (arr_z_l - 2 * arr_mid + arr_z_h) * dz_2
        )

    return laplace


@jax_backend.register_operator(CylindricalSymGrid, "gradient", rank_in=0, rank_out=1)
def make_gradient(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized gradient operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    scale_r, scale_z = 0.5 / grid.discretization

    def gradient(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        grad_r = (arr[2:, 1:-1] - arr[:-2, 1:-1]) * scale_r
        grad_z = (arr[1:-1, 2:] - arr[1:-1, :-2]) * scale_z
        # no phi dependence by definition
        return jnp.stack((grad_r, grad_z, jnp.zeros_like(grad_r)))

    return gradient


@jax_backend.register_operator(
    CylindricalSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
def make_gradient_squared(
    grid: CylindricalSymGrid, *, central: bool = True
) -> OperatorImplType:
    """Make a discretized gradient squared operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    if central:
        # use central differences
        scale_r, scale_z = 0.25 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            term_r = (arr[2:, 1:-1] - arr[:-2, 1:-1]) ** 2 * scale_r
            term_z = (arr[1:-1, 2:] - arr[1:-1, :-2]) ** 2 * scale_z
            return term_r + term_z  # type: ignore

    else:
        # use forward and backward differences
        scale_r, scale_z = 0.5 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            arr_mid = arr[1:-1, 1:-1]
            term_r = (
                (arr[2:, 1:-1] - arr_mid) ** 2 + (arr_mid - arr[:-2, 1:-1]) ** 2
            ) * scale_r
            term_z = (
                (arr[1:-1, 2:] - arr_mid) ** 2 + (arr_mid - arr[1:-1, :-2]) ** 2
            ) * scale_z
            return term_r + term_z  # type: ignore

    return gradient_squared


@jax_backend.register_operator(CylindricalSymGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized divergence operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    scale_r, scale_z = 0.5 / grid.discretization
    rs = grid.axes_coords[0]

    def divergence(arr: jax.Array) -> jax.Array:
        """Apply divergence operator to array `arr`"""
        arr_r, arr_z = arr[0], arr[1]
        return (  # type: ignore
            arr_r[1:-1, 1:-1] / rs[:, None]
            + (arr_r[2:, 1:-1] - arr_r[:-2, 1:-1]) * scale_r
            + (arr_z[1:-1, 2:] - arr_z[1:-1, :-2]) * scale_z
        )

    return divergence


@jax_backend.register_operator(
    CylindricalSymGrid, "vector_gradient", rank_in=1, rank_out=2
)
def make_vector_gradient(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized vector gradient operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    scale_r, scale_z = 0.5 / grid.discretization
    rs = grid.axes_coords[0]

    def vector_gradient(arr: jax.Array) -> jax.Array:
        """Apply vector gradient operator to array `arr`"""
        arr_r, arr_z, arr_φ = arr[0], arr[1], arr[2]

        # radial derivatives
        out_rr = (arr_r[2:, 1:-1] - arr_r[:-2, 1:-1]) * scale_r
        out_zr = (arr_z[2:, 1:-1] - arr_z[:-2, 1:-1]) * scale_r
        out_φr = (arr_φ[2:, 1:-1] - arr_φ[:-2, 1:-1]) * scale_r

        # phi-curvature terms
        out_rφ = -arr_φ[1:-1, 1:-1] / rs[:, None]
        out_φφ = arr_r[1:-1, 1:-1] / rs[:, None]
        out_zφ = jnp.zeros_like(out_rr)

        # axial derivatives
        out_rz = (arr_r[1:-1, 2:] - arr_r[1:-1, :-2]) * scale_z
        out_φz = (arr_φ[1:-1, 2:] - arr_φ[1:-1, :-2]) * scale_z
        out_zz = (arr_z[1:-1, 2:] - arr_z[1:-1, :-2]) * scale_z

        return jnp.stack(
            [
                jnp.stack([out_rr, out_rz, out_rφ]),
                jnp.stack([out_zr, out_zz, out_zφ]),
                jnp.stack([out_φr, out_φz, out_φφ]),
            ]
        )

    return vector_gradient


@jax_backend.register_operator(
    CylindricalSymGrid, "vector_laplace", rank_in=1, rank_out=1
)
def make_vector_laplace(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized vector laplace operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    dr, dz = grid.discretization
    s1, s2 = 1 / (2 * dr), 1 / dr**2
    scale_z = 1 / dz**2

    def vector_laplace(arr: jax.Array) -> jax.Array:
        """Apply vector Laplace operator to array `arr`"""
        arr_r, arr_z, arr_φ = arr[0], arr[1], arr[2]

        f_r_l = arr_r[:-2, 1:-1]
        f_r_m = arr_r[1:-1, 1:-1]
        f_r_h = arr_r[2:, 1:-1]
        out_r = (
            (arr_r[1:-1, 2:] - 2 * f_r_m + arr_r[1:-1, :-2]) * scale_z
            - f_r_m / rs[:, None] ** 2
            + (f_r_h - f_r_l) * s1 / rs[:, None]
            + (f_r_h - 2 * f_r_m + f_r_l) * s2
        )

        f_φ_l = arr_φ[:-2, 1:-1]
        f_φ_m = arr_φ[1:-1, 1:-1]
        f_φ_h = arr_φ[2:, 1:-1]
        out_φ = (
            (arr_φ[1:-1, 2:] - 2 * f_φ_m + arr_φ[1:-1, :-2]) * scale_z
            - f_φ_m / rs[:, None] ** 2
            + (f_φ_h - f_φ_l) * s1 / rs[:, None]
            + (f_φ_h - 2 * f_φ_m + f_φ_l) * s2
        )

        f_z_l = arr_z[:-2, 1:-1]
        f_z_m = arr_z[1:-1, 1:-1]
        f_z_h = arr_z[2:, 1:-1]
        out_z = (
            (arr_z[1:-1, 2:] - 2 * f_z_m + arr_z[1:-1, :-2]) * scale_z
            + (f_z_h - f_z_l) * s1 / rs[:, None]
            + (f_z_h - 2 * f_z_m + f_z_l) * s2
        )

        return jnp.stack((out_r, out_z, out_φ))

    return vector_laplace


@jax_backend.register_operator(
    CylindricalSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
def make_tensor_divergence(grid: CylindricalSymGrid) -> OperatorImplType:
    """Make a discretized tensor divergence operator for a cylindrical grid.

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    scale_r, scale_z = 0.5 / grid.discretization

    def tensor_divergence(arr: jax.Array) -> jax.Array:
        """Apply tensor divergence operator to array `arr`"""
        arr_rr, arr_rz, arr_rφ = arr[0, 0], arr[0, 1], arr[0, 2]
        arr_zr, arr_zz = arr[1, 0], arr[1, 1]
        arr_φr, arr_φz, arr_φφ = arr[2, 0], arr[2, 1], arr[2, 2]

        out_r = (
            (arr_rz[1:-1, 2:] - arr_rz[1:-1, :-2]) * scale_z
            + (arr_rr[2:, 1:-1] - arr_rr[:-2, 1:-1]) * scale_r
            + (arr_rr[1:-1, 1:-1] - arr_φφ[1:-1, 1:-1]) / rs[:, None]
        )

        out_φ = (
            (arr_φz[1:-1, 2:] - arr_φz[1:-1, :-2]) * scale_z
            + (arr_φr[2:, 1:-1] - arr_φr[:-2, 1:-1]) * scale_r
            + (arr_rφ[1:-1, 1:-1] + arr_φr[1:-1, 1:-1]) / rs[:, None]
        )

        out_z = (
            (arr_zz[1:-1, 2:] - arr_zz[1:-1, :-2]) * scale_z
            + (arr_zr[2:, 1:-1] - arr_zr[:-2, 1:-1]) * scale_r
            + arr_zr[1:-1, 1:-1] / rs[:, None]
        )

        return jnp.stack((out_r, out_z, out_φ))

    return tensor_divergence


__all__ = [
    "make_divergence",
    "make_gradient",
    "make_gradient_squared",
    "make_laplace",
    "make_tensor_divergence",
    "make_vector_gradient",
    "make_vector_laplace",
]
