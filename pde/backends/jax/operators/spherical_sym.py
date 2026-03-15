r"""This module implements differential operators on spherical grids.

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_gradient_squared
   make_divergence
   make_vector_gradient
   make_tensor_divergence
   make_tensor_double_divergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np

from .... import config
from ....grids.spherical import SphericalSymGrid
from .. import jax_backend

if TYPE_CHECKING:
    import jax

    from ....tools.typing import OperatorImplType


@jax_backend.register_operator(SphericalSymGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(
    grid: SphericalSymGrid, *, conservative: bool | None = None
) -> OperatorImplType:
    """Make a discretized laplace operator for a spherical grid.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the laplace operator should be conservative (which
            results in slightly slower computations). Conservative operators ensure mass
            conservation. If `None`, the value is read from the configuration option
            `operators.conservative_stencil`.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)
    if conservative is None:
        conservative = config["operators.conservative_stencil"]

    # calculate preliminary quantities
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]

    if conservative:
        # create a conservative spherical laplace operator
        r_min, r_max = grid.axes_bounds[0]
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        assert np.isclose(rl[0], r_min)
        assert np.isclose(rh[-1], r_max)
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (dr * volumes)
        factor_h = rh**2 / (dr * volumes)

        def laplace(arr: jax.Array) -> jax.Array:
            """Apply Laplace operator to array `arr`"""
            term_h = factor_h * (arr[2:] - arr[1:-1])
            term_l = factor_l * (arr[1:-1] - arr[:-2])
            return term_h - term_l  # type: ignore

    else:
        # create a non-conservative spherical laplace operator
        dr2 = 1 / dr**2

        def laplace(arr: jax.Array) -> jax.Array:
            """Apply Laplace operator to array `arr`"""
            diff_2 = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) * dr2
            diff_1 = (arr[2:] - arr[:-2]) / (rs * dr)
            return diff_2 + diff_1  # type: ignore

    return laplace


@jax_backend.register_operator(SphericalSymGrid, "gradient", rank_in=0, rank_out=1)
def make_gradient(
    grid: SphericalSymGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a discretized gradient operator for a spherical grid.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

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
        return jnp.stack((r, jnp.zeros_like(r), jnp.zeros_like(r)))

    return gradient


@jax_backend.register_operator(
    SphericalSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
def make_gradient_squared(
    grid: SphericalSymGrid, *, central: bool = True
) -> OperatorImplType:
    """Make a discretized gradient squared operator for a spherical grid.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

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


@jax_backend.register_operator(SphericalSymGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(
    grid: SphericalSymGrid,
    *,
    conservative: bool | None = None,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a discretized divergence operator for a spherical grid.

    Warning:
        This operator ignores the θ-component of the field when calculating the
        divergence. This is because the resulting scalar field could not be expressed
        on a :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation. If `None`, the value is read from the configuration option
            `operators.conservative_stencil`.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)
    if conservative is None:
        conservative = config["operators.conservative_stencil"]

    # calculate preliminary quantities
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]

    if conservative:
        # implement conservative version of the divergence operator
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (2 * volumes)
        factor_h = rh**2 / (2 * volumes)

        def divergence(arr: jax.Array) -> jax.Array:
            """Apply divergence operator to array `arr`"""
            arr_r = arr[0]
            if method == "central":
                term_h = factor_h * (arr_r[1:-1] + arr_r[2:])
                term_l = factor_l * (arr_r[:-2] + arr_r[1:-1])
            elif method == "forward":
                term_h = 2 * factor_h * arr_r[2:]
                term_l = 2 * factor_l * arr_r[1:-1]
            elif method == "backward":
                term_h = 2 * factor_h * arr_r[1:-1]
                term_l = 2 * factor_l * arr_r[:-2]
            return term_h - term_l  # type: ignore

    else:
        # implement naive divergence operator
        factors = 2 / rs  # factors that need to be multiplied

        def divergence(arr: jax.Array) -> jax.Array:
            """Apply divergence operator to array `arr`"""
            arr_r = arr[0]
            if method == "central":
                diff_r = (arr_r[2:] - arr_r[:-2]) / (2 * dr)
            elif method == "forward":
                diff_r = (arr_r[2:] - arr_r[1:-1]) / dr
            elif method == "backward":
                diff_r = (arr_r[1:-1] - arr_r[:-2]) / dr
            return diff_r + factors * arr_r[1:-1]  # type: ignore

    return divergence


@jax_backend.register_operator(
    SphericalSymGrid, "vector_gradient", rank_in=1, rank_out=2
)
def make_vector_gradient(
    grid: SphericalSymGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a discretized vector gradient operator for a spherical grid.

    Warning:
        This operator ignores the two angular components of the field when calculating
        the gradient. This is because the resulting field could not be expressed on a
        :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    if method == "central":
        scale_r = 0.5 / grid.discretization[0]
    elif method in {"forward", "backward"}:
        scale_r = 1 / grid.discretization[0]
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def vector_gradient(arr: jax.Array) -> jax.Array:
        """Apply vector gradient operator to array `arr`"""
        arr_r = arr[0]

        if method == "central":
            out_rr = (arr_r[2:] - arr_r[:-2]) * scale_r
        elif method == "forward":
            out_rr = (arr_r[2:] - arr_r[1:-1]) * scale_r
        elif method == "backward":
            out_rr = (arr_r[1:-1] - arr_r[:-2]) * scale_r

        out_diag = arr_r[1:-1] / rs
        zeros = jnp.zeros_like(out_rr)

        return jnp.stack(
            [
                jnp.stack([out_rr, zeros, zeros]),
                jnp.stack([zeros, out_diag, zeros]),
                jnp.stack([zeros, zeros, out_diag]),
            ]
        )

    return vector_gradient


@jax_backend.register_operator(
    SphericalSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
def make_tensor_divergence(
    grid: SphericalSymGrid,
    *,
    conservative: bool | None = False,
) -> OperatorImplType:
    """Make a discretized tensor divergence operator for a spherical grid.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation. If `None`, the value is read from the configuration option
            `operators.conservative_stencil`.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)
    if conservative is None:
        conservative = config["operators.conservative_stencil"]

    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]

    if conservative:
        # conservative implementation of the tensor divergence
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (2 * volumes)
        factor_h = rh**2 / (2 * volumes)
        area_factor = (rh**2 - rl**2) / volumes

        def tensor_divergence(arr: jax.Array) -> jax.Array:
            """Apply tensor divergence operator to array `arr`"""
            arr_rr = arr[0, 0]
            arr_φφ = arr[2, 2]

            term_r_h = factor_h * (arr_rr[1:-1] + arr_rr[2:])
            term_r_l = factor_l * (arr_rr[:-2] + arr_rr[1:-1])
            out_r = term_r_h - term_r_l - area_factor * arr_φφ[1:-1]
            zeros = jnp.zeros_like(out_r)

            return jnp.stack([out_r, zeros, zeros])

    else:
        # naive implementation of the tensor divergence
        scale_r = 1 / (2 * dr)

        def tensor_divergence(arr: jax.Array) -> jax.Array:
            """Apply tensor divergence operator to array `arr`"""
            arr_rr = arr[0, 0]
            arr_rφ = arr[0, 2]
            arr_θr = arr[1, 0]
            arr_φr = arr[2, 0]
            arr_φφ = arr[2, 2]

            out_r = (arr_rr[2:] - arr_rr[:-2]) * scale_r + 2 * (
                arr_rr[1:-1] - arr_φφ[1:-1]
            ) / rs
            out_θ = (arr_θr[2:] - arr_θr[:-2]) * scale_r + 2 * arr_θr[1:-1] / rs
            out_φ = (arr_φr[2:] - arr_φr[:-2]) * scale_r + (
                2 * arr_φr[1:-1] + arr_rφ[1:-1]
            ) / rs

            return jnp.stack([out_r, out_θ, out_φ])

    return tensor_divergence


@jax_backend.register_operator(
    SphericalSymGrid, "tensor_double_divergence", rank_in=2, rank_out=0
)
def make_tensor_double_divergence(
    grid: SphericalSymGrid,
    *,
    conservative: bool | None = None,
) -> OperatorImplType:
    """Make a discretized tensor double divergence operator for a spherical grid.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The spherical grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation. If `None`, the value is read from the configuration option
            `operators.conservative_stencil`.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)
    if conservative is None:
        conservative = config["operators.conservative_stencil"]

    # calculate preliminary quantities
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]

    if conservative:
        # create a conservative double divergence laplace operator
        r_min, r_max = grid.axes_bounds[0]
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        assert np.isclose(rl[0], r_min)
        assert np.isclose(rh[-1], r_max)
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl / volumes
        factor_h = rh / volumes
        factor2_l = rl**2 / (dr * volumes)
        factor2_h = rh**2 / (dr * volumes)

        def tensor_double_divergence(arr: jax.Array) -> jax.Array:
            """Apply double divergence operator to tensor array `arr`"""
            arr_rr = arr[0, 0]
            arr_φφ = arr[2, 2]

            # radial part
            arr_rr_h = arr_rr[1:-1] + arr_rr[2:]
            arr_rr_l = arr_rr[:-2] + arr_rr[1:-1]
            arr_rr_dr_h = arr_rr[2:] - arr_rr[1:-1]
            arr_rr_dr_l = arr_rr[1:-1] - arr_rr[:-2]
            div2_rr_h = factor_h * arr_rr_h + factor2_h * arr_rr_dr_h
            div2_rr_l = factor_l * arr_rr_l + factor2_l * arr_rr_dr_l
            div2_rr = div2_rr_h - div2_rr_l

            # angular part
            arr_φφ_h = arr_φφ[1:-1] + arr_φφ[2:]
            arr_φφ_l = arr_φφ[:-2] + arr_φφ[1:-1]
            div2_φφ = factor_h * arr_φφ_h - factor_l * arr_φφ_l

            return div2_rr - div2_φφ  # type: ignore

    else:
        # naive, non-conservative implementation of the double divergence
        dr2 = 1 / dr**2
        scale_r = 1 / (2 * dr)

        def tensor_double_divergence(arr: jax.Array) -> jax.Array:
            """Apply double divergence operator to tensor array `arr`"""
            arr_rr = arr[0, 0]
            arr_φφ = arr[2, 2]

            # first derivatives
            arr_rr_dr = (arr_rr[2:] - arr_rr[:-2]) * scale_r
            arr_φφ_dr = (arr_φφ[2:] - arr_φφ[:-2]) * scale_r

            # second derivative
            term1 = (arr_rr[2:] - arr_rr[:-2]) / (rs * dr)
            term2 = (arr_rr[2:] - 2 * arr_rr[1:-1] + arr_rr[:-2]) * dr2
            lap_rr = term1 + term2

            enum = (arr_rr[1:-1] - arr_φφ[1:-1]) / rs + arr_rr_dr - arr_φφ_dr
            return lap_rr + 2 * enum / rs  # type: ignore

    return tensor_double_divergence


__all__ = [
    "make_divergence",
    "make_gradient",
    "make_gradient_squared",
    "make_laplace",
    "make_tensor_divergence",
    "make_tensor_double_divergence",
    "make_vector_gradient",
]
