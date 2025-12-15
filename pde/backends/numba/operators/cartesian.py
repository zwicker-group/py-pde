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

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

from .... import config
from ....grids.cartesian import CartesianGrid
from ....tools.misc import module_available
from .. import numba_backend
from ..utils import jit

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....tools.typing import NumericArray, OperatorImplType


def make_corner_point_setter_2d(grid: CartesianGrid) -> Callable[[NumericArray], None]:
    """Make a helper function that sets the virtual corner points of a 2d field.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the helper function is created

    Returns:
        A function that can be applied to an array of values
    """
    periodic_x, periodic_y = grid.periodic

    @jit
    def set_corner_points(arr: NumericArray) -> None:
        """Set the corner points of the array `arr`"""
        if periodic_x:
            # exploit periodicity along x-direction to use known boundary points
            arr[0, 0] = arr[-2, 0]
            arr[-1, 0] = arr[1, 0]
            arr[0, -1] = arr[-2, -1]
            arr[-1, -1] = arr[1, -1]

        elif periodic_y:
            # exploit periodicity along y-direction to use known boundary points
            arr[0, 0] = arr[0, -2]
            arr[-1, 0] = arr[-1, 1]
            arr[0, -1] = arr[0, -2]
            arr[-1, -1] = arr[-1, 1]

        else:
            # we cannot exploit any periodicity, so we interpolate
            arr[0, 0] = 0.5 * (arr[0, 1] + arr[1, 0])
            arr[-1, 0] = 0.5 * (arr[-1, 1] + arr[-2, 0])
            arr[0, -1] = 0.5 * (arr[0, -2] + arr[1, -1])
            arr[-1, -1] = 0.5 * (arr[-1, -2] + arr[-2, -1])

    return set_corner_points  # type: ignore


def _make_laplace_numba_1d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 1d Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale: float = grid.discretization[0] ** -2

    @jit
    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i - 1] = (arr[i - 1] - 2 * arr[i] + arr[i + 1]) * scale

    return laplace  # type: ignore


def _make_laplace_numba_2d(
    grid: CartesianGrid, *, corner_weight: float | None = None
) -> OperatorImplType:
    """Make a 2d Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        corner_weight (float):
            Weighting factor for corner points of stencil. If `None`, value is read from
            the configuration option `operators.cartesian_laplacian_2d_corner_weight`.
            The standard value is zero, which corresponds to the traditional 5-point
            stencil. Typical alternative choices are 1/2 (Oono-Puri stencil) and 1/3
            (Patra-Karttunen or Mehrstellen stencil); see
            https://en.wikipedia.org/wiki/Nine-point_stencil.

    Returns:
        A function that can be applied to an array of values
    """
    if corner_weight is None:
        corner_weight = config["operators.cartesian.laplacian_2d_corner_weight"]

    # use parallel processing for large enough arrays
    dim_x, dim_y = grid.shape
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    if corner_weight == 0:
        # use standard 5-point stencil
        scale_x, scale_y = grid.discretization**-2

        @jit(parallel=parallel)
        def laplace(arr: NumericArray, out: NumericArray) -> None:
            """Apply Laplace operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    lap_x = (arr[i - 1, j] - 2 * arr[i, j] + arr[i + 1, j]) * scale_x
                    lap_y = (arr[i, j - 1] - 2 * arr[i, j] + arr[i, j + 1]) * scale_y
                    out[i - 1, j - 1] = lap_x + lap_y

    else:
        # use 9-point stencil with interpolated boundary conditions
        w = corner_weight
        numba_backend._logger.info(
            "Create 2D Cartesian Laplacian with 9-point stencil (w=%.3g)", w
        )

        if not np.isclose(*grid.discretization):
            # we have not yet found a good expression for the 9-point stencil for dx!=dy
            numba_backend._logger.warning(
                "9-point stencils with anisotropic grids are not tested and might "
                "produce wrong results."
            )

        # prepare the stencil matrix
        dxm2, dym2 = grid.discretization**-2
        dm2 = dxm2 + dym2
        stencil = np.array(
            [
                [0.25 * dm2 * w, dxm2 * (1 - w), 0.25 * dm2 * w],
                [dym2 * (1 - w), (dxm2 + dym2) * (w - 2), dym2 * (1 - w)],
                [0.25 * dm2 * w, dxm2 * (1 - w), 0.25 * dm2 * w],
            ]
        )

        set_corner_points = make_corner_point_setter_2d(grid)

        @jit(parallel=parallel)
        def laplace(arr: NumericArray, out: NumericArray) -> None:
            """Apply Laplace operator to array `arr`"""
            set_corner_points(arr)
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    value = 0
                    for x in range(3):
                        for y in range(3):
                            value += arr[i + x - 1, j + y - 1] * stencil[x, y]
                    out[i - 1, j - 1] = value

    return laplace  # type: ignore


def _make_laplace_numba_3d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 3d Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = grid.discretization**-2

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    val_mid = 2 * arr[i, j, k]
                    lap_x = (arr[i - 1, j, k] - val_mid + arr[i + 1, j, k]) * scale_x
                    lap_y = (arr[i, j - 1, k] - val_mid + arr[i, j + 1, k]) * scale_y
                    lap_z = (arr[i, j, k - 1] - val_mid + arr[i, j, k + 1]) * scale_z
                    out[i - 1, j - 1, k - 1] = lap_x + lap_y + lap_z

    return laplace  # type: ignore


def _make_laplace_numba_spectral_1d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 1d spectral Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import fft

    assert module_available("rocket_fft")
    assert grid.periodic[0]  # we currently only support periodic grid
    ks = 2 * np.pi * fft.fftfreq(grid.shape[0], grid.discretization[0])
    factor = -(ks**2)

    @register_jitable
    def laplace_impl(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        out[:] = fft.ifft(factor * fft.fft(arr[1:-1]))

    @overload(laplace_impl)
    def ol_laplace(arr: NumericArray, out: NumericArray):
        """Integrates data over a grid using numba."""
        if np.isrealobj(arr):
            # special case of a real array

            def laplace_real(arr: NumericArray, out: NumericArray) -> None:
                """Apply Laplace operator to array `arr`"""
                out[:] = fft.ifft(factor * fft.fft(arr[1:-1])).real

            return laplace_real

        return laplace_impl

    @jit
    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        laplace_impl(arr, out)

    return laplace  # type: ignore


def _make_laplace_numba_spectral_2d(grid: CartesianGrid) -> OperatorImplType:
    """Make a 2d spectral Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import fft

    assert module_available("rocket_fft")
    assert all(grid.periodic)  # we currently only support periodic grid
    ks = [fft.fftfreq(grid.shape[i], grid.discretization[i]) for i in range(2)]
    factor = -4 * np.pi**2 * (ks[0][:, None] ** 2 + ks[1][None, :] ** 2)

    @register_jitable
    def laplace_impl(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        out[:] = fft.ifft2(factor * fft.fft2(arr[1:-1, 1:-1]))

    @overload(laplace_impl)
    def ol_laplace(arr: NumericArray, out: NumericArray):
        """Integrates data over a grid using numba."""
        if np.isrealobj(arr):
            # special case of a real array

            def laplace_real(arr: NumericArray, out: NumericArray) -> None:
                """Apply Laplace operator to array `arr`"""
                out[:] = fft.ifft2(factor * fft.fft2(arr[1:-1, 1:-1])).real

            return laplace_real

        return laplace_impl

    @jit
    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        laplace_impl(arr, out)

    return laplace  # type: ignore


@numba_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(
    grid: CartesianGrid, *, spectral: bool | None = None, **kwargs
) -> OperatorImplType:
    """Make a Laplace operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        spectral (bool or None):
            Flag deciding whether a spectral implementation is used. If `None`, the
            value is controlled by the configuration.
        **kwargs:
            Specifies extra arguments influencing how the operator is created. Note that

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    if spectral is None:
        spectral = config["operators.cartesian.default_backend"] == "numba-spectral"

    if spectral:
        # use spectral versions of the operators
        if dim == 1:
            return _make_laplace_numba_spectral_1d(grid, **kwargs)
        if dim == 2:
            return _make_laplace_numba_spectral_2d(grid, **kwargs)
        msg = f"Numba spectral Laplace operator not implemented for {dim:d} dimensions"
        raise NotImplementedError(msg)

    # use finite-difference operators
    if dim == 1:
        return _make_laplace_numba_1d(grid, **kwargs)
    if dim == 2:
        return _make_laplace_numba_2d(grid, **kwargs)
    if dim == 3:
        return _make_laplace_numba_3d(grid, **kwargs)
    msg = f"Numba Laplace operator not implemented for {dim:d} dimensions"
    raise NotImplementedError(msg)


def _make_gradient_numba_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 1d gradient operator using numba compilation.

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

    dim_x = grid.shape[0]
    dx = grid.discretization[0]

    @jit
    def gradient(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            if method == "central":
                out[0, i - 1] = (arr[i + 1] - arr[i - 1]) / (2 * dx)
            elif method == "forward":
                out[0, i - 1] = (arr[i + 1] - arr[i]) / dx
            elif method == "backward":
                out[0, i - 1] = (arr[i] - arr[i - 1]) / dx

    return gradient  # type: ignore


def _make_gradient_numba_2d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 2d gradient operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    if method == "central":
        scale_x, scale_y = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                if method == "central":
                    out[0, i - 1, j - 1] = (arr[i + 1, j] - arr[i - 1, j]) * scale_x
                    out[1, i - 1, j - 1] = (arr[i, j + 1] - arr[i, j - 1]) * scale_y
                elif method == "forward":
                    out[0, i - 1, j - 1] = (arr[i + 1, j] - arr[i, j]) * scale_x
                    out[1, i - 1, j - 1] = (arr[i, j + 1] - arr[i, j]) * scale_y
                elif method == "backward":
                    out[0, i - 1, j - 1] = (arr[i, j] - arr[i - 1, j]) * scale_x
                    out[1, i - 1, j - 1] = (arr[i, j] - arr[i, j - 1]) * scale_y

    return gradient  # type: ignore


def _make_gradient_numba_3d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 3d gradient operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    if method == "central":
        scale_x, scale_y, scale_z = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y, scale_z = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    if method == "central":
                        out[0, i - 1, j - 1, k - 1] = (
                            arr[i + 1, j, k] - arr[i - 1, j, k]
                        ) * scale_x
                        out[1, i - 1, j - 1, k - 1] = (
                            arr[i, j + 1, k] - arr[i, j - 1, k]
                        ) * scale_y
                        out[2, i - 1, j - 1, k - 1] = (
                            arr[i, j, k + 1] - arr[i, j, k - 1]
                        ) * scale_z
                    elif method == "forward":
                        out[0, i - 1, j - 1, k - 1] = (
                            arr[i + 1, j, k] - arr[i, j, k]
                        ) * scale_x
                        out[1, i - 1, j - 1, k - 1] = (
                            arr[i, j + 1, k] - arr[i, j, k]
                        ) * scale_y
                        out[2, i - 1, j - 1, k - 1] = (
                            arr[i, j, k + 1] - arr[i, j, k]
                        ) * scale_z
                    elif method == "backward":
                        out[0, i - 1, j - 1, k - 1] = (
                            arr[i, j, k] - arr[i - 1, j, k]
                        ) * scale_x
                        out[1, i - 1, j - 1, k - 1] = (
                            arr[i, j, k] - arr[i, j - 1, k]
                        ) * scale_y
                        out[2, i - 1, j - 1, k - 1] = (
                            arr[i, j, k] - arr[i, j, k - 1]
                        ) * scale_z

    return gradient  # type: ignore


@numba_backend.register_operator(CartesianGrid, "gradient", rank_in=0, rank_out=1)
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
        return _make_gradient_numba_1d(grid, method=method)
    if dim == 2:
        return _make_gradient_numba_2d(grid, method=method)
    if dim == 3:
        return _make_gradient_numba_3d(grid, method=method)
    msg = f"Numba gradient operator not implemented for dimension {dim}"
    raise NotImplementedError(msg)


def _make_gradient_squared_numba_1d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 1d squared gradient operator using numba compilation.

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
    dim_x = grid.shape[0]

    if central:
        # use central differences
        scale = 0.25 / grid.discretization[0] ** 2

        @jit
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / grid.discretization[0] ** 2

        @jit
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                diff_l = (arr[i + 1] - arr[i]) ** 2
                diff_r = (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = (diff_l + diff_r) * scale

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_2d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 2d squared gradient operator using numba compilation.

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
    dim_x, dim_y = grid.shape

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    if central:
        # use central differences
        scale_x, scale_y = 0.25 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    term_x = (arr[i + 1, j] - arr[i - 1, j]) ** 2 * scale_x
                    term_y = (arr[i, j + 1] - arr[i, j - 1]) ** 2 * scale_y
                    out[i - 1, j - 1] = term_x + term_y

    else:
        # use forward and backward differences
        scale_x, scale_y = 0.5 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    term_x = (
                        (arr[i + 1, j] - arr[i, j]) ** 2
                        + (arr[i, j] - arr[i - 1, j]) ** 2
                    ) * scale_x
                    term_y = (
                        (arr[i, j + 1] - arr[i, j]) ** 2
                        + (arr[i, j] - arr[i, j - 1]) ** 2
                    ) * scale_y
                    out[i - 1, j - 1] = term_x + term_y

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_3d(
    grid: CartesianGrid, central: bool = True
) -> OperatorImplType:
    """Make a 3d squared gradient operator using numba compilation.

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
    dim_x, dim_y, dim_z = grid.shape

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    if central:
        # use central differences
        scale_x, scale_y, scale_z = 0.25 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    for k in range(1, dim_z + 1):
                        term_x = (arr[i + 1, j, k] - arr[i - 1, j, k]) ** 2 * scale_x
                        term_y = (arr[i, j + 1, k] - arr[i, j - 1, k]) ** 2 * scale_y
                        term_z = (arr[i, j, k + 1] - arr[i, j, k - 1]) ** 2 * scale_z
                        out[i - 1, j - 1, k - 1] = term_x + term_y + term_z

    else:
        # use forward and backward differences
        scale_x, scale_y, scale_z = 0.5 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    for k in range(1, dim_z + 1):
                        term_x = (
                            (arr[i + 1, j, k] - arr[i, j, k]) ** 2
                            + (arr[i, j, k] - arr[i - 1, j, k]) ** 2
                        ) * scale_x
                        term_y = (
                            (arr[i, j + 1, k] - arr[i, j, k]) ** 2
                            + (arr[i, j, k] - arr[i, j - 1, k]) ** 2
                        ) * scale_y
                        term_z = (
                            (arr[i, j, k + 1] - arr[i, j, k]) ** 2
                            + (arr[i, j, k] - arr[i, j, k - 1]) ** 2
                        ) * scale_z
                        out[i - 1, j - 1, k - 1] = term_x + term_y + term_z

    return gradient_squared  # type: ignore


@numba_backend.register_operator(
    CartesianGrid, "gradient_squared", rank_in=0, rank_out=0
)
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
        return _make_gradient_squared_numba_1d(grid, central=central)
    if dim == 2:
        return _make_gradient_squared_numba_2d(grid, central=central)
    if dim == 3:
        return _make_gradient_squared_numba_3d(grid, central=central)
    msg = f"Squared gradient operator is not implemented for dimension {dim}"
    raise NotImplementedError(msg)


def _make_divergence_numba_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 1d divergence operator using numba compilation.

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
    dim_x = grid.shape[0]
    dx = grid.discretization[0]

    @jit
    def divergence(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            if method == "central":
                out[i - 1] = (arr[0, i + 1] - arr[0, i - 1]) / (2 * dx)
            elif method == "forward":
                out[i - 1] = (arr[0, i + 1] - arr[0, i]) / dx
            elif method == "backward":
                out[i - 1] = (arr[0, i] - arr[0, i - 1]) / dx

    return divergence  # type: ignore


def _make_divergence_numba_2d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 2d divergence operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    if method == "central":
        scale_x, scale_y = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                if method == "central":
                    d_x = (arr[0, i + 1, j] - arr[0, i - 1, j]) * scale_x
                    d_y = (arr[1, i, j + 1] - arr[1, i, j - 1]) * scale_y
                elif method == "forward":
                    d_x = (arr[0, i + 1, j] - arr[0, i, j]) * scale_x
                    d_y = (arr[1, i, j + 1] - arr[1, i, j]) * scale_y
                elif method == "backward":
                    d_x = (arr[0, i, j] - arr[0, i - 1, j]) * scale_x
                    d_y = (arr[1, i, j] - arr[1, i, j - 1]) * scale_y
                out[i - 1, j - 1] = d_x + d_y

    return divergence  # type: ignore


def _make_divergence_numba_3d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a 3d divergence operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    if method == "central":
        scale_x, scale_y, scale_z = 0.5 / grid.discretization
    elif method in {"forward", "backward"}:
        scale_x, scale_y, scale_z = 1 / grid.discretization
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    if method == "central":
                        d_x = (arr[0, i + 1, j, k] - arr[0, i - 1, j, k]) * scale_x
                        d_y = (arr[1, i, j + 1, k] - arr[1, i, j - 1, k]) * scale_y
                        d_z = (arr[2, i, j, k + 1] - arr[2, i, j, k - 1]) * scale_z
                    elif method == "forward":
                        d_x = (arr[0, i + 1, j, k] - arr[0, i, j, k]) * scale_x
                        d_y = (arr[1, i, j + 1, k] - arr[1, i, j, k]) * scale_y
                        d_z = (arr[2, i, j, k + 1] - arr[2, i, j, k]) * scale_z
                    elif method == "backward":
                        d_x = (arr[0, i, j, k] - arr[0, i - 1, j, k]) * scale_x
                        d_y = (arr[1, i, j, k] - arr[1, i, j - 1, k]) * scale_y
                        d_z = (arr[2, i, j, k] - arr[2, i, j, k - 1]) * scale_z
                    out[i - 1, j - 1, k - 1] = d_x + d_y + d_z

    return divergence  # type: ignore


@numba_backend.register_operator(CartesianGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
    """Make a divergence operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the divergence operator. The value is read from
            the configuration for `config`, and a suitable backend is chosen for `auto`.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if dim == 1:
        return _make_divergence_numba_1d(grid, method=method)
    if dim == 2:
        return _make_divergence_numba_2d(grid, method=method)
    if dim == 3:
        return _make_divergence_numba_3d(grid, method=method)
    msg = f"Numba divergence operator not implemented for dimension {dim}"
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
        backend (str):
            Backend used for calculating the vectorized operator.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    operator = make_operator(grid, **kwargs)

    def vectorized_operator(arr: NumericArray, out: NumericArray) -> None:
        """Apply vector gradient operator to array `arr`"""
        for i in range(dim):
            operator(arr[i], out[i])

    return register_jitable(vectorized_operator)  # type: ignore


@numba_backend.register_operator(
    CartesianGrid, "vector_gradient", rank_in=1, rank_out=2
)
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


@numba_backend.register_operator(CartesianGrid, "vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(grid: CartesianGrid) -> OperatorImplType:
    """Make a vector Laplacian on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_laplace, grid)


@numba_backend.register_operator(
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
