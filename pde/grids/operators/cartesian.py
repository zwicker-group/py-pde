"""This module implements differential operators on Cartesian grids.

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_vector_laplace
   make_tensor_divergence
   make_poisson_solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

from ... import config
from ...tools.misc import module_available
from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..boundaries.axes import BoundariesBase, BoundariesList
from ..boundaries.axis import BoundaryAxisBase
from ..cartesian import CartesianGrid
from .common import make_general_poisson_solver, uniform_discretization

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


def make_corner_point_setter_2d(grid: CartesianGrid) -> Callable[[np.ndarray], None]:
    """Make a helper function that sets the virtual corner points of a 2d field.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the helper function is created

    Returns:
        A function that can be applied to an array of values
    """
    periodic_x, periodic_y = grid.periodic

    @jit
    def set_corner_points(arr: np.ndarray) -> None:
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


def _get_laplace_matrix_1d(bcs: BoundariesList) -> tuple[np.ndarray, np.ndarray]:
    """Get sparse matrix for Laplace operator on a 1d Cartesian grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    dim_x = bcs.grid.shape[0]
    matrix = sparse.dok_matrix((dim_x, dim_x))
    vector = sparse.dok_matrix((dim_x, 1))

    for i in range(dim_x):
        matrix[i, i] += -2

        if i == 0:
            const, entries = bcs[0].get_sparse_matrix_data((-1,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i - 1] += 1

        if i == dim_x - 1:
            const, entries = bcs[0].get_sparse_matrix_data((dim_x,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i + 1] += 1

    matrix *= bcs.grid.discretization[0] ** -2
    vector *= bcs.grid.discretization[0] ** -2

    return matrix, vector


def _get_laplace_matrix_2d(bcs: BoundariesList) -> tuple[np.ndarray, np.ndarray]:
    """Get sparse matrix for Laplace operator on a 2d Cartesian grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    dim_x, dim_y = bcs.grid.shape
    matrix = sparse.dok_matrix((dim_x * dim_y, dim_x * dim_y))
    vector = sparse.dok_matrix((dim_x * dim_y, 1))

    bc_x, bc_y = bcs
    scale_x, scale_y = bcs.grid.discretization**-2

    def i(x, y):
        """Helper function for flattening the index.

        This is equivalent to np.ravel_multi_index((x, y), (dim_x, dim_y))
        """
        return x * dim_y + y

    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_x + scale_y))

    for x in range(dim_x):
        for y in range(dim_y):
            # handle x-direction
            if x == 0:
                const, entries = bc_x.get_sparse_matrix_data((-1, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x - 1, y)] += scale_x

            if x == dim_x - 1:
                const, entries = bc_x.get_sparse_matrix_data((dim_x, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x + 1, y)] += scale_x

            # handle y-direction
            if y == 0:
                const, entries = bc_y.get_sparse_matrix_data((x, -1))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y - 1)] += scale_y

            if y == dim_y - 1:
                const, entries = bc_y.get_sparse_matrix_data((x, dim_y))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y + 1)] += scale_y

    return matrix, vector


def _get_laplace_matrix_3d(bcs: BoundariesList) -> tuple[np.ndarray, np.ndarray]:
    """Get sparse matrix for Laplace operator on a 3d Cartesian grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    dim_x, dim_y, dim_z = bcs.grid.shape
    matrix = sparse.dok_matrix((dim_x * dim_y * dim_z, dim_x * dim_y * dim_z))
    vector = sparse.dok_matrix((dim_x * dim_y * dim_z, 1))

    bc_x, bc_y, bc_z = bcs
    scale_x, scale_y, scale_z = bcs.grid.discretization**-2

    def i(x, y, z):
        """Helper function for flattening the index.

        This is equivalent to np.ravel_multi_index((x, y, z), (dim_x, dim_y, dim_z))
        """
        return (x * dim_y + y) * dim_z + z

    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_x + scale_y + scale_z))

    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                # handle x-direction
                if x == 0:
                    const, entries = bc_x.get_sparse_matrix_data((-1, y, z))
                    vector[i(x, y, z)] += const * scale_x
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(k, y, z)] += v * scale_x
                else:
                    matrix[i(x, y, z), i(x - 1, y, z)] += scale_x

                if x == dim_x - 1:
                    const, entries = bc_x.get_sparse_matrix_data((dim_x, y, z))
                    vector[i(x, y, z)] += const * scale_x
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(k, y, z)] += v * scale_x
                else:
                    matrix[i(x, y, z), i(x + 1, y, z)] += scale_x

                # handle y-direction
                if y == 0:
                    const, entries = bc_y.get_sparse_matrix_data((x, -1, z))
                    vector[i(x, y, z)] += const * scale_y
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(x, k, z)] += v * scale_y
                else:
                    matrix[i(x, y, z), i(x, y - 1, z)] += scale_y

                if y == dim_y - 1:
                    const, entries = bc_y.get_sparse_matrix_data((x, dim_y, z))
                    vector[i(x, y, z)] += const * scale_y
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(x, k, z)] += v * scale_y
                else:
                    matrix[i(x, y, z), i(x, y + 1, z)] += scale_y

                # handle z-direction
                if z == 0:
                    const, entries = bc_z.get_sparse_matrix_data((x, y, -1))
                    vector[i(x, y, z)] += const * scale_z
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(x, y, k)] += v * scale_z
                else:
                    matrix[i(x, y, z), i(x, y, z - 1)] += scale_z

                if z == dim_z - 1:
                    const, entries = bc_z.get_sparse_matrix_data((x, y, dim_z))
                    vector[i(x, y, z)] += const * scale_z
                    for k, v in entries.items():
                        matrix[i(x, y, z), i(x, y, k)] += v * scale_z
                else:
                    matrix[i(x, y, z), i(x, y, z + 1)] += scale_z

    return matrix, vector


def _get_laplace_matrix(bcs: BoundariesList) -> tuple[np.ndarray, np.ndarray]:
    """Get sparse matrix for Laplace operator on a Cartesian grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    dim = bcs.grid.dim

    if dim == 1:
        result = _get_laplace_matrix_1d(bcs)
    elif dim == 2:
        result = _get_laplace_matrix_2d(bcs)
    elif dim == 3:
        result = _get_laplace_matrix_3d(bcs)
    else:
        raise NotImplementedError(f"{dim:d}-dimensional Laplace matrix not implemented")

    return result


def _make_laplace_scipy_nd(grid: CartesianGrid) -> OperatorType:
    """Make a Laplace operator using the scipy module.

    This only supports uniform discretizations.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = uniform_discretization(grid) ** -2

    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        assert arr.shape == grid._shape_full
        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            out[:] = ndimage.laplace(scaling * arr)[valid]

    return laplace


def _make_laplace_numba_1d(grid: CartesianGrid) -> OperatorType:
    """Make a 1d Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale: float = grid.discretization[0] ** -2  # type: ignore

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i - 1] = (arr[i - 1] - 2 * arr[i] + arr[i + 1]) * scale

    return laplace  # type: ignore


def _make_laplace_numba_2d(
    grid: CartesianGrid, *, corner_weight: float | None = None
) -> OperatorType:
    """Make a 2d Laplace operator using numba compilation.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        corner_weight (float):
            Weighting factor for the corner points of the stencil. If `None`, the value
            is read from the configuration option `operators.cartesian_laplacian_2d_corner_weight`.
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
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
            """Apply Laplace operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    lap_x = (arr[i - 1, j] - 2 * arr[i, j] + arr[i + 1, j]) * scale_x
                    lap_y = (arr[i, j - 1] - 2 * arr[i, j] + arr[i, j + 1]) * scale_y
                    out[i - 1, j - 1] = lap_x + lap_y

    else:
        # use 9-point stencil with interpolated boundary conditions
        w = corner_weight
        _logger.info("Create 2D Cartesian Laplacian with 9-point stencil (w=%.3g)", w)

        if not np.isclose(*grid.discretization):
            # we have not yet found a good expression for the 9-point stencil for dx!=dy
            _logger.warning(
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
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
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


def _make_laplace_numba_3d(grid: CartesianGrid) -> OperatorType:
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
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
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


def _make_laplace_numba_spectral_1d(grid: CartesianGrid) -> OperatorType:
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
    def laplace_impl(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        out[:] = fft.ifft(factor * fft.fft(arr[1:-1]))

    @overload(laplace_impl)
    def ol_laplace(arr: np.ndarray, out: np.ndarray):
        """Integrates data over a grid using numba."""
        if np.isrealobj(arr):
            # special case of a real array

            def laplace_real(arr: np.ndarray, out: np.ndarray) -> None:
                """Apply Laplace operator to array `arr`"""
                out[:] = fft.ifft(factor * fft.fft(arr[1:-1])).real

            return laplace_real

        else:
            return laplace_impl

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        laplace_impl(arr, out)

    return laplace  # type: ignore


def _make_laplace_numba_spectral_2d(grid: CartesianGrid) -> OperatorType:
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
    def laplace_impl(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        out[:] = fft.ifft2(factor * fft.fft2(arr[1:-1, 1:-1]))

    @overload(laplace_impl)
    def ol_laplace(arr: np.ndarray, out: np.ndarray):
        """Integrates data over a grid using numba."""
        if np.isrealobj(arr):
            # special case of a real array

            def laplace_real(arr: np.ndarray, out: np.ndarray) -> None:
                """Apply Laplace operator to array `arr`"""
                out[:] = fft.ifft2(factor * fft.fft2(arr[1:-1, 1:-1])).real

            return laplace_real

        else:
            return laplace_impl

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply Laplace operator to array `arr`"""
        laplace_impl(arr, out)

    return laplace  # type: ignore


@CartesianGrid.register_operator("laplace", rank_in=0, rank_out=0)
def make_laplace(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "numba-spectral", "scipy"] = "config",
    **kwargs,
) -> OperatorType:
    """Make a Laplace operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the Laplace operator. The value is read from
            the configuration for `config`, and a suitable backend is chosen for `auto`.
        **kwargs:
            Specifies extra arguments influencing how the operator is created. Note that

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if backend == "config":  # read value from configuration
        backend = config["operators.cartesian.default_backend"]

    if backend == "auto":  # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            laplace = _make_laplace_numba_1d(grid, **kwargs)
        elif dim == 2:
            laplace = _make_laplace_numba_2d(grid, **kwargs)
        elif dim == 3:
            laplace = _make_laplace_numba_3d(grid, **kwargs)
        else:
            raise NotImplementedError(
                f"Numba Laplace operator not implemented for {dim:d} dimensions"
            )

    elif backend == "numba-spectral":
        if dim == 1:
            laplace = _make_laplace_numba_spectral_1d(grid, **kwargs)
        elif dim == 2:
            laplace = _make_laplace_numba_spectral_2d(grid, **kwargs)
        else:
            raise NotImplementedError(
                f"Spectral Laplace operator not implemented for {dim:d} dimensions"
            )

    elif backend == "scipy":
        laplace = _make_laplace_scipy_nd(grid, **kwargs)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return laplace


def _make_gradient_scipy_nd(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorType:
    """Make a gradient operator using the scipy module.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = 1 / grid.discretization
    dim = grid.dim
    shape_out = (dim,) + grid.shape

    if method == "central":
        stencil = [-0.5, 0, 0.5]
    elif method == "forward":
        stencil = [0, -1, 1]
    elif method == "backward":
        stencil = [-1, 1, 0]
    else:
        raise ValueError(f"Unknown derivative type `{method}`")

    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply gradient operator to array `arr`"""
        assert arr.shape == grid._shape_full
        if out is None:
            out = np.empty(shape_out)
        else:
            assert out.shape == shape_out

        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            for i in range(dim):
                out[i] = ndimage.correlate1d(arr, stencil, axis=i)[valid] * scaling[i]

    return gradient


def _make_gradient_numba_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")

    dim_x = grid.shape[0]
    dx = grid.discretization[0]

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
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
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
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
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
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


@CartesianGrid.register_operator("gradient", rank_in=0, rank_out=1)
def make_gradient(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
    """Make a gradient operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the gradient operator. The value is read from
            the configuration for `config`, and a suitable backend is chosen for `auto`.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if backend == "config":  # read value from configuration
        backend = config["operators.cartesian.default_backend"]

    if backend == "auto":  # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            gradient = _make_gradient_numba_1d(grid, method=method)
        elif dim == 2:
            gradient = _make_gradient_numba_2d(grid, method=method)
        elif dim == 3:
            gradient = _make_gradient_numba_3d(grid, method=method)
        else:
            raise NotImplementedError(
                f"Numba gradient operator not implemented for dimension {dim}"
            )

    elif backend == "scipy":
        gradient = _make_gradient_scipy_nd(grid, method=method)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return gradient


def _make_gradient_squared_numba_1d(
    grid: CartesianGrid, central: bool = True
) -> OperatorType:
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
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / grid.discretization[0] ** 2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                diff_l = (arr[i + 1] - arr[i]) ** 2
                diff_r = (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = (diff_l + diff_r) * scale

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_2d(
    grid: CartesianGrid, central: bool = True
) -> OperatorType:
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
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
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
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
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
) -> OperatorType:
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
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
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
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
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


@CartesianGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
def make_gradient_squared(grid: CartesianGrid, *, central: bool = True) -> OperatorType:
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
        gradient_squared = _make_gradient_squared_numba_1d(grid, central=central)
    elif dim == 2:
        gradient_squared = _make_gradient_squared_numba_2d(grid, central=central)
    elif dim == 3:
        gradient_squared = _make_gradient_squared_numba_3d(grid, central=central)
    else:
        raise NotImplementedError(
            f"Squared gradient operator is not implemented for dimension {dim}"
        )

    return gradient_squared


def _make_divergence_scipy_nd(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorType:
    """Make a divergence operator using the scipy module.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    data_shape = grid._shape_full
    scale = 1 / grid.discretization
    if method == "central":
        stencil = [-0.5, 0, 0.5]
    elif method == "forward":
        stencil = [0, -1, 1]
    elif method == "backward":
        stencil = [-1, 1, 0]
    else:
        raise ValueError(f"Unknown derivative type `{method}`")

    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply divergence operator to array `arr`"""
        assert arr.shape[0] == len(data_shape)
        assert arr.shape[1:] == data_shape

        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(grid.shape, dtype=arr.dtype)
        else:
            out[:] = 0

        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            for i in range(len(data_shape)):
                out += ndimage.correlate1d(arr[i], stencil, axis=i)[valid] * scale[i]

    return divergence


def _make_divergence_numba_1d(
    grid: CartesianGrid, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")
    dim_x = grid.shape[0]
    dx = grid.discretization[0]

    @jit
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
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
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
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
) -> OperatorType:
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
        raise ValueError(f"Unknown derivative type `{method}`")

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
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


@CartesianGrid.register_operator("divergence", rank_in=1, rank_out=0)
def make_divergence(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
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

    if backend == "config":  # read value from configuration
        backend = config["operators.cartesian.default_backend"]

    if backend == "auto":  # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            divergence = _make_divergence_numba_1d(grid, method=method)
        elif dim == 2:
            divergence = _make_divergence_numba_2d(grid, method=method)
        elif dim == 3:
            divergence = _make_divergence_numba_3d(grid, method=method)
        else:
            raise NotImplementedError(
                f"Numba divergence operator not implemented for dimension {dim}"
            )

    elif backend == "scipy":
        divergence = _make_divergence_scipy_nd(grid, method=method)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return divergence


def _vectorize_operator(
    make_operator: Callable,
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    **kwargs,
) -> OperatorType:
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
    operator = make_operator(grid, backend=backend, **kwargs)

    def vectorized_operator(arr: np.ndarray, out: np.ndarray) -> None:
        """Apply vector gradient operator to array `arr`"""
        for i in range(dim):
            operator(arr[i], out[i])

    if backend == "numba":
        return register_jitable(vectorized_operator)  # type: ignore
    else:
        return vectorized_operator


@CartesianGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
def make_vector_gradient(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
    """Make a vector gradient operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector gradient operator. The value is read
            from the configuration for `config`, and a suitable backend is chosen for
            `auto`.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_gradient, grid, backend=backend, method=method)


@CartesianGrid.register_operator("vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
) -> OperatorType:
    """Make a vector Laplacian on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector Laplace operator.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_laplace, grid, backend=backend)


@CartesianGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
def make_tensor_divergence(
    grid: CartesianGrid,
    *,
    backend: Literal["auto", "config", "numba", "scipy"] = "config",
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
    """Make a tensor divergence operator on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the tensor divergence operator.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_divergence, grid, backend=backend, method=method)


@CartesianGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
def make_poisson_solver(
    bcs: BoundariesList, *, method: Literal["auto", "scipy"] = "auto"
) -> OperatorType:
    """Make a operator that solves Poisson's equation.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str):
            Method used for calculating the tensor divergence operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)


__all__ = [
    "make_laplace",
    "make_gradient",
    "make_divergence",
    "make_vector_gradient",
    "make_vector_laplace",
    "make_tensor_divergence",
    "make_poisson_solver",
]
