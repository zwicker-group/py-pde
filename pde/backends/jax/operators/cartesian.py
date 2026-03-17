"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_vector_laplace
   make_tensor_divergence

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from ....grids.cartesian import CartesianGrid
from .. import jax_backend

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from ....tools.typing import OperatorImplType


def _make_laplace_jax_1d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 1d Laplace operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    scale: float = grid.discretization[0] ** -2

    def laplace(arr: jax.Array) -> jax.Array:
        """Apply Laplace operator to array `arr`"""
        return scale * (arr[0:-2] + arr[2:] - 2 * arr[1:-1])

    return laplace


def _make_laplace_jax_2d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 2d Laplace operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    scale_x, scale_y = grid.discretization**-2

    def laplace(arr: jax.Array) -> jax.Array:
        """Apply Laplace operator to array `arr`"""
        lap_x = scale_x * (arr[0:-2, 1:-1] + arr[2:, 1:-1] - 2 * arr[1:-1, 1:-1])
        lap_y = scale_y * (arr[1:-1, 0:-2] + arr[1:-1, 2:] - 2 * arr[1:-1, 1:-1])
        return lap_x + lap_y  # type: ignore

    return laplace


def _make_laplace_jax_3d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 3d Laplace operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    scale_x, scale_y, scale_z = grid.discretization**-2

    def laplace(arr: jax.Array) -> jax.Array:
        """Apply Laplace operator to array `arr`"""
        lap_x = arr[0:-2, 1:-1, 1:-1] + arr[2:, 1:-1, 1:-1] - 2 * arr[1:-1, 1:-1, 1:-1]
        lap_y = arr[1:-1, 0:-2, 1:-1] + arr[1:-1, 2:, 1:-1] - 2 * arr[1:-1, 1:-1, 1:-1]
        lap_z = arr[1:-1, 1:-1, 0:-2] + arr[1:-1, 1:-1, 2:] - 2 * arr[1:-1, 1:-1, 1:-1]
        return scale_x * lap_x + scale_y * lap_y + scale_z * lap_z  # type: ignore

    return laplace


@jax_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(grid: CartesianGrid, **kwargs) -> OperatorImplType:
    """Make a Laplace operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        **kwargs:
            Specifies extra arguments influencing how the operator is created.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    # use finite-difference operators
    if dim == 1:
        return _make_laplace_jax_1d(grid, **kwargs)
    if dim == 2:
        return _make_laplace_jax_2d(grid, **kwargs)
    if dim == 3:
        return _make_laplace_jax_3d(grid, **kwargs)
    msg = f"Jax Laplace operator not implemented for {dim:d} dimensions"
    raise NotImplementedError(msg)


def _make_gradient_jax_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 1d gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method not in {"central", "forward", "backward"}:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    dx = grid.discretization[0]

    def gradient(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            grad_x = (arr[2:] - arr[:-2]) / (2 * dx)
        elif method == "forward":
            grad_x = (arr[2:] - arr[1:-1]) / dx
        elif method == "backward":
            grad_x = (arr[1:-1] - arr[:-2]) / dx
        else:
            raise RuntimeError
        return grad_x[None, :]  # type: ignore

    return gradient


def _make_gradient_jax_2d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 2d gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method == "central":
        scale_x, scale_y = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def gradient(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            grad_x = (arr[2:, 1:-1] - arr[:-2, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 2:] - arr[1:-1, :-2]) * scale_y
        elif method == "forward":
            grad_x = (arr[2:, 1:-1] - arr[1:-1, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 2:] - arr[1:-1, 1:-1]) * scale_y
        elif method == "backward":
            grad_x = (arr[1:-1, 1:-1] - arr[:-2, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 1:-1] - arr[1:-1, :-2]) * scale_y
        return jnp.stack((grad_x, grad_y))

    return gradient


def _make_gradient_jax_3d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 3d gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method == "central":
        scale_x, scale_y, scale_z = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y, scale_z = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def gradient(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            grad_x = (arr[2:, 1:-1, 1:-1] - arr[:-2, 1:-1, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 2:, 1:-1] - arr[1:-1, :-2, 1:-1]) * scale_y
            grad_z = (arr[1:-1, 1:-1, 2:] - arr[1:-1, 1:-1, :-2]) * scale_z
        elif method == "forward":
            grad_x = (arr[2:, 1:-1, 1:-1] - arr[1:-1, 1:-1, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 2:, 1:-1] - arr[1:-1, 1:-1, 1:-1]) * scale_y
            grad_z = (arr[1:-1, 1:-1, 2:] - arr[1:-1, 1:-1, 1:-1]) * scale_z
        elif method == "backward":
            grad_x = (arr[1:-1, 1:-1, 1:-1] - arr[:-2, 1:-1, 1:-1]) * scale_x
            grad_y = (arr[1:-1, 1:-1, 1:-1] - arr[1:-1, :-2, 1:-1]) * scale_y
            grad_z = (arr[1:-1, 1:-1, 1:-1] - arr[1:-1, 1:-1, :-2]) * scale_z
        return jnp.stack((grad_x, grad_y, grad_z))

    return gradient


@jax_backend.register_operator(CartesianGrid, "gradient", rank_in=0, rank_out=1)
def make_gradient(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a gradient operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if dim == 1:
        return _make_gradient_jax_1d(grid, method=method)
    if dim == 2:
        return _make_gradient_jax_2d(grid, method=method)
    if dim == 3:
        return _make_gradient_jax_3d(grid, method=method)
    msg = f"Jax gradient operator not implemented for dimension {dim}"
    raise NotImplementedError(msg)


def _make_gradient_squared_jax_1d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 1d squared gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
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
        scale = 0.25 / grid.discretization[0] ** 2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            return (arr[2:] - arr[:-2]) ** 2 * scale  # type: ignore

    else:
        # use forward and backward differences
        scale = 0.5 / grid.discretization[0] ** 2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            diff_l = (arr[2:] - arr[1:-1]) ** 2
            diff_r = (arr[1:-1] - arr[:-2]) ** 2
            return (diff_l + diff_r) * scale  # type: ignore

    return gradient_squared


def _make_gradient_squared_jax_2d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 2d squared gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
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
        scale_x, scale_y = 0.25 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            term_x = (arr[2:, 1:-1] - arr[:-2, 1:-1]) ** 2 * scale_x
            term_y = (arr[1:-1, 2:] - arr[1:-1, :-2]) ** 2 * scale_y
            return term_x + term_y  # type: ignore

    else:
        # use forward and backward differences
        scale_x, scale_y = 0.5 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            term_x = (
                (arr[2:, 1:-1] - arr[1:-1, 1:-1]) ** 2
                + (arr[1:-1, 1:-1] - arr[:-2, 1:-1]) ** 2
            ) * scale_x
            term_y = (
                (arr[1:-1, 2:] - arr[1:-1, 1:-1]) ** 2
                + (arr[1:-1, 1:-1] - arr[1:-1, :-2]) ** 2
            ) * scale_y
            return term_x + term_y  # type: ignore

    return gradient_squared


def _make_gradient_squared_jax_3d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 3d squared gradient operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
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
        scale_x, scale_y, scale_z = 0.25 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            term_x = (arr[2:, 1:-1, 1:-1] - arr[:-2, 1:-1, 1:-1]) ** 2 * scale_x
            term_y = (arr[1:-1, 2:, 1:-1] - arr[1:-1, :-2, 1:-1]) ** 2 * scale_y
            term_z = (arr[1:-1, 1:-1, 2:] - arr[1:-1, 1:-1, :-2]) ** 2 * scale_z
            return term_x + term_y + term_z  # type: ignore

    else:
        # use forward and backward differences
        scale_x, scale_y, scale_z = 0.5 / grid.discretization**2

        def gradient_squared(arr: jax.Array) -> jax.Array:
            """Apply squared gradient operator to array `arr`"""
            term_x = (
                (arr[2:, 1:-1, 1:-1] - arr[1:-1, 1:-1, 1:-1]) ** 2
                + (arr[1:-1, 1:-1, 1:-1] - arr[:-2, 1:-1, 1:-1]) ** 2
            ) * scale_x
            term_y = (
                (arr[1:-1, 2:, 1:-1] - arr[1:-1, 1:-1, 1:-1]) ** 2
                + (arr[1:-1, 1:-1, 1:-1] - arr[1:-1, :-2, 1:-1]) ** 2
            ) * scale_y
            term_z = (
                (arr[1:-1, 1:-1, 2:] - arr[1:-1, 1:-1, 1:-1]) ** 2
                + (arr[1:-1, 1:-1, 1:-1] - arr[1:-1, 1:-1, :-2]) ** 2
            ) * scale_z
            return term_x + term_y + term_z  # type: ignore

    return gradient_squared


@jax_backend.register_operator(CartesianGrid, "gradient_squared", rank_in=0, rank_out=0)
def make_gradient_squared(
    grid: CartesianGrid, *, central: bool = True
) -> OperatorImplType:
    """Make a gradient operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if dim == 1:
        return _make_gradient_squared_jax_1d(grid, central=central)
    if dim == 2:
        return _make_gradient_squared_jax_2d(grid, central=central)
    if dim == 3:
        return _make_gradient_squared_jax_3d(grid, central=central)
    msg = f"Squared gradient operator is not implemented for dimension {dim}"
    raise NotImplementedError(msg)


def _make_divergence_jax_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 1d divergence operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method not in {"central", "forward", "backward"}:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)
    dx = grid.discretization[0]

    def divergence(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            return (arr[0, 2:] - arr[0, :-2]) / (2 * dx)  # type: ignore
        if method == "forward":
            return (arr[0, 2:] - arr[0, 1:-1]) / dx  # type: ignore
        if method == "backward":
            return (arr[0, 1:-1] - arr[0, :-2]) / dx  # type: ignore
        # this cannot be reached because we validated the method before
        assert False  # noqa: B011, PT015

    return divergence


def _make_divergence_jax_2d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 2d divergence operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method == "central":
        scale_x, scale_y = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def divergence(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            d_x = (arr[0, 2:, 1:-1] - arr[0, :-2, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 2:] - arr[1, 1:-1, :-2]) * scale_y
        elif method == "forward":
            d_x = (arr[0, 2:, 1:-1] - arr[0, 1:-1, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 2:] - arr[1, 1:-1, 1:-1]) * scale_y
        elif method == "backward":
            d_x = (arr[0, 1:-1, 1:-1] - arr[0, :-2, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 1:-1] - arr[1, 1:-1, :-2]) * scale_y
        return d_x + d_y  # type: ignore

    return divergence


def _make_divergence_jax_3d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 3d divergence operator using jax compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    if method == "central":
        scale_x, scale_y, scale_z = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y, scale_z = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def divergence(arr: jax.Array) -> jax.Array:
        """Apply gradient operator to array `arr`"""
        if method == "central":
            d_x = (arr[0, 2:, 1:-1, 1:-1] - arr[0, :-2, 1:-1, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 2:, 1:-1] - arr[1, 1:-1, :-2, 1:-1]) * scale_y
            d_z = (arr[2, 1:-1, 1:-1, 2:] - arr[2, 1:-1, 1:-1, :-2]) * scale_z
        elif method == "forward":
            d_x = (arr[0, 2:, 1:-1, 1:-1] - arr[0, 1:-1, 1:-1, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 2:, 1:-1] - arr[1, 1:-1, 1:-1, 1:-1]) * scale_y
            d_z = (arr[2, 1:-1, 1:-1, 2:] - arr[2, 1:-1, 1:-1, 1:-1]) * scale_z
        elif method == "backward":
            d_x = (arr[0, 1:-1, 1:-1, 1:-1] - arr[0, :-2, 1:-1, 1:-1]) * scale_x
            d_y = (arr[1, 1:-1, 1:-1, 1:-1] - arr[1, 1:-1, :-2, 1:-1]) * scale_y
            d_z = (arr[2, 1:-1, 1:-1, 1:-1] - arr[2, 1:-1, 1:-1, :-2]) * scale_z
        return d_x + d_y + d_z  # type: ignore

    return divergence


@jax_backend.register_operator(CartesianGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a divergence operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if dim == 1:
        return _make_divergence_jax_1d(grid, method=method)
    if dim == 2:
        return _make_divergence_jax_2d(grid, method=method)
    if dim == 3:
        return _make_divergence_jax_3d(grid, method=method)
    msg = f"Jax divergence operator not implemented for dimension {dim}"
    raise NotImplementedError(msg)


def _vectorize_operator(
    make_operator: Callable, grid: CartesianGrid, **kwargs
) -> OperatorImplType:
    """Apply an operator to on all dimensions of a vector.

    Args:
        make_operator (callable):
            The function that creates the basic operator
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        **kwargs:
            Additional keyword arguments passed to the operator

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    operator = make_operator(grid, **kwargs)

    def vectorized_operator(arr: jax.Array) -> jax.Array:
        """Apply vector gradient operator to array `arr`"""
        return jnp.stack([operator(arr[i]) for i in range(dim)])

    return vectorized_operator


@jax_backend.register_operator(CartesianGrid, "vector_gradient", rank_in=1, rank_out=2)
def make_vector_gradient(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a vector gradient operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_gradient, grid, method=method)


@jax_backend.register_operator(CartesianGrid, "vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(grid: CartesianGrid) -> OperatorImplType:
    """Make a vector Laplacian on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_laplace, grid)


@jax_backend.register_operator(
    CartesianGrid, "tensor_divergence", rank_in=2, rank_out=1
)
def make_tensor_divergence(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a tensor divergence operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_divergence, grid, method=method)


__all__ = [
    "make_divergence",
    "make_gradient",
    "make_laplace",
    "make_tensor_divergence",
    "make_vector_gradient",
    "make_vector_laplace",
]
