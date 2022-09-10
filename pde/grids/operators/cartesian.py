"""
This module implements differential operators on Cartesian grids 

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

from typing import Callable, Tuple

import numba as nb
import numpy as np
from numba.extending import register_jitable

from ... import config
from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..boundaries import Boundaries
from ..cartesian import CartesianGrid
from .common import make_general_poisson_solver, uniform_discretization


def _get_laplace_matrix_1d(bcs: Boundaries) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 1d Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
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
            const, entries = bcs[0].get_data((-1,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i - 1] += 1

        if i == dim_x - 1:
            const, entries = bcs[0].get_data((dim_x,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i + 1] += 1

    matrix *= bcs.grid.discretization[0] ** -2
    vector *= bcs.grid.discretization[0] ** -2

    return matrix, vector


def _get_laplace_matrix_2d(bcs: Boundaries) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 2d Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
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
        """helper function for flattening the index

        This is equivalent to np.ravel_multi_index((x, y), (dim_x, dim_y))
        """
        return x * dim_y + y

    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_x + scale_y))

    for x in range(dim_x):
        for y in range(dim_y):
            # handle x-direction
            if x == 0:
                const, entries = bc_x.get_data((-1, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x - 1, y)] += scale_x

            if x == dim_x - 1:
                const, entries = bc_x.get_data((dim_x, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x + 1, y)] += scale_x

            # handle y-direction
            if y == 0:
                const, entries = bc_y.get_data((x, -1))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y - 1)] += scale_y

            if y == dim_y - 1:
                const, entries = bc_y.get_data((x, dim_y))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y + 1)] += scale_y

    return matrix, vector


def _get_laplace_matrix(bcs: Boundaries) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 1d Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
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
    else:
        raise NotImplementedError(f"{dim:d}-dimensional Laplace matrix not implemented")

    return result


def _make_derivative(
    grid: CartesianGrid, axis: int = 0, method: str = "central"
) -> OperatorType:
    """make a derivative operator along a single axis using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken
        method (str):
            The method for calculating the derivative. Possible values are
            'central', 'forward', and 'backward'.

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown derivative type `{method}`")

    shape = grid.shape
    dim = len(shape)
    dx = grid.discretization[axis]

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if dim == 1:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                if method == "central":
                    out[i - 1] = (arr[i + 1] - arr[i - 1]) / (2 * dx)
                elif method == "forward":
                    out[i - 1] = (arr[i + 1] - arr[i]) / dx
                elif method == "backward":
                    out[i - 1] = (arr[i] - arr[i - 1]) / dx

    elif dim == 2:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    if method == "central":
                        out[i - 1, j - 1] = (arr_r - arr_l) / (2 * dx)
                    elif method == "forward":
                        out[i - 1, j - 1] = (arr_r - arr[i, j]) / dx
                    elif method == "backward":
                        out[i - 1, j - 1] = (arr[i, j] - arr_l) / dx

    elif dim == 3:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        if method == "central":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr_l) / (2 * dx)
                        elif method == "forward":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr[i, j, k]) / dx
                        elif method == "backward":
                            out[i - 1, j - 1, k - 1] = (arr[i, j, k] - arr_l) / dx

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {dim:d} dimensions"
        )

    return diff  # type: ignore


def _make_derivative2(grid: CartesianGrid, axis: int = 0) -> OperatorType:
    """make a second-order derivative operator along a single axis

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    shape = grid.shape
    dim = len(shape)
    scale = 1 / grid.discretization[axis] ** 2

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if dim == 1:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * scale

    elif dim == 2:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    out[i - 1, j - 1] = (arr_r - 2 * arr[i, j] + arr_l) * scale

    elif dim == 3:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        out[i - 1, j - 1, k - 1] = (
                            arr_r - 2 * arr[i, j, k] + arr_l
                        ) * scale

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {dim:d} dimensions"
        )

    return diff  # type: ignore


def _make_laplace_scipy_nd(grid: CartesianGrid) -> OperatorType:
    """make a laplace operator using the scipy module

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
        """apply laplace operator to array `arr`"""
        assert arr.shape == grid._shape_full
        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            out[:] = ndimage.laplace(scaling * arr)[valid]

    return laplace


def _make_laplace_numba_1d(grid: CartesianGrid) -> OperatorType:
    """make a 1d laplace operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = grid.discretization[0] ** -2

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply laplace operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i - 1] = (arr[i - 1] - 2 * arr[i] + arr[i + 1]) * scale

    return laplace  # type: ignore


def _make_laplace_numba_2d(grid: CartesianGrid) -> OperatorType:
    """make a 2d laplace operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = grid.discretization**-2

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply laplace operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                lap_x = (arr[i - 1, j] - 2 * arr[i, j] + arr[i + 1, j]) * scale_x
                lap_y = (arr[i, j - 1] - 2 * arr[i, j] + arr[i, j + 1]) * scale_y
                out[i - 1, j - 1] = lap_x + lap_y

    return laplace  # type: ignore


def _make_laplace_numba_3d(grid: CartesianGrid) -> OperatorType:
    """make a 3d laplace operator using numba compilation

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
        """apply laplace operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    val_mid = 2 * arr[i, j, k]
                    lap_x = (arr[i - 1, j, k] - val_mid + arr[i + 1, j, k]) * scale_x
                    lap_y = (arr[i, j - 1, k] - val_mid + arr[i, j + 1, k]) * scale_y
                    lap_z = (arr[i, j, k - 1] - val_mid + arr[i, j, k + 1]) * scale_z
                    out[i - 1, j - 1, k - 1] = lap_x + lap_y + lap_z

    return laplace  # type: ignore


@CartesianGrid.register_operator("laplace", rank_in=0, rank_out=0)
def make_laplace(grid: CartesianGrid, backend: str = "auto") -> OperatorType:
    """make a laplace operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the laplace operator. If backend='auto', a
            suitable backend is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if backend == "auto":
        # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            laplace = _make_laplace_numba_1d(grid)
        elif dim == 2:
            laplace = _make_laplace_numba_2d(grid)
        elif dim == 3:
            laplace = _make_laplace_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba laplace operator not implemented for {dim:d} dimensions"
            )

    # elif backend == "matrix":
    #     laplace = make_laplace_from_matrix(*_get_laplace_matrix(grid))

    elif backend == "scipy":
        laplace = _make_laplace_scipy_nd(grid)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return laplace


def _make_gradient_scipy_nd(grid: CartesianGrid) -> OperatorType:
    """make a gradient operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = 0.5 / grid.discretization
    dim = grid.dim
    shape_out = (dim,) + grid.shape

    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        assert arr.shape == grid._shape_full
        if out is None:
            out = np.empty(shape_out)
        else:
            assert out.shape == shape_out

        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            for i in range(dim):
                out[i] = ndimage.convolve1d(arr, [1, 0, -1], axis=i)[valid] * scaling[i]

    return gradient


def _make_gradient_numba_1d(grid: CartesianGrid) -> OperatorType:
    """make a 1d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = 0.5 / grid.discretization[0]

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[0, i - 1] = (arr[i + 1] - arr[i - 1]) * scale

    return gradient  # type: ignore


def _make_gradient_numba_2d(grid: CartesianGrid) -> OperatorType:
    """make a 2d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                out[0, i - 1, j - 1] = (arr[i + 1, j] - arr[i - 1, j]) * scale_x
                out[1, i - 1, j - 1] = (arr[i, j + 1] - arr[i, j - 1]) * scale_y

    return gradient  # type: ignore


def _make_gradient_numba_3d(grid: CartesianGrid) -> OperatorType:
    """make a 3d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    out[0, i - 1, j - 1, k - 1] = (
                        arr[i + 1, j, k] - arr[i - 1, j, k]
                    ) * scale_x
                    out[1, i - 1, j - 1, k - 1] = (
                        arr[i, j + 1, k] - arr[i, j - 1, k]
                    ) * scale_y
                    out[2, i - 1, j - 1, k - 1] = (
                        arr[i, j, k + 1] - arr[i, j, k - 1]
                    ) * scale_z

    return gradient  # type: ignore


@CartesianGrid.register_operator("gradient", rank_in=0, rank_out=1)
def make_gradient(grid: CartesianGrid, backend: str = "auto") -> OperatorType:
    """make a gradient operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the gradient operator.
            If backend='auto', a suitable backend is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if backend == "auto":
        # choose the fastest available gradient operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            gradient = _make_gradient_numba_1d(grid)
        elif dim == 2:
            gradient = _make_gradient_numba_2d(grid)
        elif dim == 3:
            gradient = _make_gradient_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba gradient operator not implemented for dimension {dim}"
            )

    elif backend == "scipy":
        gradient = _make_gradient_scipy_nd(grid)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return gradient


def _make_gradient_squared_numba_1d(
    grid: CartesianGrid, central: bool = True
) -> OperatorType:
    """make a 1d squared gradient operator using numba compilation

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
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / grid.discretization[0] ** 2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                diff_l = (arr[i + 1] - arr[i]) ** 2
                diff_r = (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = (diff_l + diff_r) * scale

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_2d(
    grid: CartesianGrid, central: bool = True
) -> OperatorType:
    """make a 2d squared gradient operator using numba compilation

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
            """apply squared gradient operator to array `arr`"""
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
            """apply squared gradient operator to array `arr`"""
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
    """make a 3d squared gradient operator using numba compilation

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
            """apply squared gradient operator to array `arr`"""
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
            """apply squared gradient operator to array `arr`"""
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
def make_gradient_squared(grid: CartesianGrid, central: bool = True) -> OperatorType:
    """make a gradient operator on a Cartesian grid

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


def _make_divergence_scipy_nd(grid: CartesianGrid) -> OperatorType:
    """make a divergence operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    data_shape = grid._shape_full
    scale = 0.5 / grid.discretization

    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply divergence operator to array `arr`"""
        assert arr.shape[0] == len(data_shape) and arr.shape[1:] == data_shape

        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(grid.shape, dtype=arr.dtype)
        else:
            out[:] = 0

        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            for i in range(len(data_shape)):
                out += ndimage.convolve1d(arr[i], [1, 0, -1], axis=i)[valid] * scale[i]

    return divergence


def _make_divergence_numba_1d(grid: CartesianGrid) -> OperatorType:
    """make a 1d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = 0.5 / grid.discretization[0]

    @jit
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i - 1] = (arr[0, i + 1] - arr[0, i - 1]) * scale

    return divergence  # type: ignore


def _make_divergence_numba_2d(grid: CartesianGrid) -> OperatorType:
    """make a 2d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                d_x = (arr[0, i + 1, j] - arr[0, i - 1, j]) * scale_x
                d_y = (arr[1, i, j + 1] - arr[1, i, j - 1]) * scale_y
                out[i - 1, j - 1] = d_x + d_y

    return divergence  # type: ignore


def _make_divergence_numba_3d(grid: CartesianGrid) -> OperatorType:
    """make a 3d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    d_x = (arr[0, i + 1, j, k] - arr[0, i - 1, j, k]) * scale_x
                    d_y = (arr[1, i, j + 1, k] - arr[1, i, j - 1, k]) * scale_y
                    d_z = (arr[2, i, j, k + 1] - arr[2, i, j, k - 1]) * scale_z
                    out[i - 1, j - 1, k - 1] = d_x + d_y + d_z

    return divergence  # type: ignore


@CartesianGrid.register_operator("divergence", rank_in=1, rank_out=0)
def make_divergence(grid: CartesianGrid, backend: str = "auto") -> OperatorType:
    """make a divergence operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the divergence operator.
            If backend='auto', a suitable backend is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if backend == "auto":
        # choose the fastest available divergence operator
        if 1 <= dim <= 3:
            backend = "numba"
        else:
            backend = "scipy"

    if backend == "numba":
        if dim == 1:
            divergence = _make_divergence_numba_1d(grid)
        elif dim == 2:
            divergence = _make_divergence_numba_2d(grid)
        elif dim == 3:
            divergence = _make_divergence_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba divergence operator not implemented for dimension {dim}"
            )

    elif backend == "scipy":
        divergence = _make_divergence_scipy_nd(grid)

    else:
        raise ValueError(f"Backend `{backend}` is not defined")

    return divergence


def _vectorize_operator(
    make_operator: Callable, grid: CartesianGrid, *, backend: str = "numba"
) -> OperatorType:
    """apply an operator to on all dimensions of a vector

    Args:
        make_operator (callable):
            The function that creates the basic operator
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector gradient operator.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    operator = make_operator(grid, backend=backend)

    def vectorized_operator(arr: np.ndarray, out: np.ndarray) -> None:
        """apply vector gradient operator to array `arr`"""
        for i in range(dim):
            operator(arr[i], out[i])

    if backend == "numba":
        return register_jitable(vectorized_operator)  # type: ignore
    else:
        return vectorized_operator


@CartesianGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
def make_vector_gradient(grid: CartesianGrid, backend: str = "numba") -> OperatorType:
    """make a vector gradient operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector gradient operator.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_gradient, grid, backend=backend)


@CartesianGrid.register_operator("vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(grid: CartesianGrid, backend: str = "numba") -> OperatorType:
    """make a vector Laplacian on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector laplace operator.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_laplace, grid, backend=backend)


@CartesianGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
def make_tensor_divergence(grid: CartesianGrid, backend: str = "numba") -> OperatorType:
    """make a tensor divergence operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the tensor divergence operator.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_divergence, grid, backend=backend)


@CartesianGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
def make_poisson_solver(bcs: Boundaries, method: str = "auto") -> OperatorType:
    """make a operator that solves Poisson's equation

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
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
