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
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>   
"""

from typing import Tuple

import numba as nb
import numpy as np

from ... import config
from ...tools.numba import jit, jit_allocate_out
from ...tools.typing import OperatorType
from ..cartesian import CartesianGridBase
from .common import make_general_poisson_solver, uniform_discretization


def _get_laplace_matrix_1d(grid: CartesianGridBase) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 1d Cartesian grid

    Args:
        grid (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    dim_x = grid.shape[0]
    matrix = sparse.dok_matrix((dim_x, dim_x))
    vector = sparse.dok_matrix((dim_x, 1))

    for i in range(1, dim_x + 1):
        matrix[i, i] += -2

        if i == 0:
            const, entries = grid[0].get_data((-1,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i - 1] += 1

        if i == dim_x - 1:
            const, entries = grid[0].get_data((dim_x,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i + 1] += 1

    matrix *= grid.discretization[0] ** -2
    vector *= grid.discretization[0] ** -2

    return matrix, vector


def _get_laplace_matrix_2d(grid: CartesianGridBase) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 2d Cartesian grid

    Args:
        grid (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    dim_x, dim_y = grid.shape
    matrix = sparse.dok_matrix((dim_x * dim_y, dim_x * dim_y))
    vector = sparse.dok_matrix((dim_x * dim_y, 1))

    bc_x, bc_y = grid
    scale_x, scale_y = grid.discretization ** -2

    def i(x, y):
        """helper function for flattening the index

        This is equivalent to np.ravel_multi_index((x, y), (dim_x, dim_y))
        """
        return x * dim_y + y

    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_x + scale_y))

    for x in range(1, dim_x + 1):
        for y in range(1, dim_y + 1):
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


def _get_laplace_matrix(grid: CartesianGridBase) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a 1d Cartesian grid

    Args:
        grid (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    dim = grid.dim
    grid.check_value_rank(0)

    if dim == 1:
        result = _get_laplace_matrix_1d(grid)
    elif dim == 2:
        result = _get_laplace_matrix_2d(grid)
    else:
        raise NotImplementedError(
            f"Numba laplace operator not implemented for {dim:d} dimensions"
        )

    return result


def _make_derivative(
    grid: CartesianGridBase, axis: int = 0, method: str = "central"
) -> OperatorType:
    """make a derivative operator along a single axis using numba compilation

    Args:
        grid (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        axis (int):
            The axis along which the derivative will be taken
        method (str):
            The method for calculating the derivative. Possible values are
            'central', 'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values. The result will be
        an array of the same shape containing the actual derivatives at the grid
        points.
    """
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown derivative type `{method}`")

    shape = grid.shape
    dim = len(shape)
    dx = grid.discretization[axis]
    region = grid[axis].make_region_evaluator()

    if dim == 1:

        @jit_allocate_out(out_shape=shape)
        def diff(arr: np.ndarray, out: np.ndarray = None) -> np.ndarray:
            """calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                if method == "central":
                    out[i] = (arr[i + 1] - arr[i - 1]) * 0.5 / dx
                elif method == "forward":
                    out[i] = (arr[i + 1] - arr[i]) / dx
                elif method == "backward":
                    out[i] = (arr[i] - arr[i - 1]) / dx

            return out

    elif dim == 2:
        # TODO: generalize this

        @jit_allocate_out(out_shape=shape)
        def diff(arr: np.ndarray, out: np.ndarray = None) -> np.ndarray:
            """calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l, arr_m, arr_h = region(arr, (i, j))
                    if method == "central":
                        out[i, j] = (arr_h - arr_l) * 0.5 / dx
                    elif method == "forward":
                        out[i, j] = (arr_h - arr_m) / dx
                    elif method == "backward":
                        out[i, j] = (arr_m - arr_l) / dx

            return out

    elif dim == 3:

        @jit_allocate_out(out_shape=shape)
        def diff(arr: np.ndarray, out: np.ndarray = None) -> np.ndarray:
            """calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l, arr_m, arr_h = region(arr, (i, j, k))
                        if method == "central":
                            out[i, j, k] = (arr_h - arr_l) * 0.5 / dx
                        elif method == "forward":
                            out[i, j, k] = (arr_h - arr_m) / dx
                        elif method == "backward":
                            out[i, j, k] = (arr_m - arr_l) / dx

            return out

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {dim:d} dimensions"
        )

    return diff  # type: ignore


def _make_laplace_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a laplace operator using the scipy module

    This only supports uniform discretizations.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = uniform_discretization(grid) ** -2

    def laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply laplace operator to array `arr`"""
        assert arr.shape == tuple(s + 2 for s in grid.shape)
        return ndimage.laplace(scaling * arr, output=out)

    return laplace


def _make_laplace_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d laplace operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = grid.discretization[0] ** -2

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply laplace operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i] = (arr[i - 1] - 2 * arr[i] + arr[i + 1]) * scale

        return out

    return laplace  # type: ignore


def _make_laplace_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d laplace operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = grid.discretization ** -2

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply laplace operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                lap_x = (arr[i - 1, j] - 2 * arr[i, j] + arr[i + 1, j]) * scale_x
                lap_y = (arr[i, j - 1] - 2 * arr[i, j] + arr[i, j + 1]) * scale_y
                out[i, j] = lap_x + lap_y

        return out

    return laplace  # type: ignore


def _make_laplace_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d laplace operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = grid.discretization ** -2

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply laplace operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    val_mid = 2 * arr[i, j, k]
                    lap_x = (arr[i - 1, j, k] - val_mid + arr[i + 1, j, k]) * scale_x
                    lap_y = (arr[i, j - 1, k] - val_mid + arr[i, j + 1, k]) * scale_y
                    lap_z = (arr[i, j, k - 1] - val_mid + arr[i, j, k + 1]) * scale_z
                    out[i, j, k] = lap_x + lap_y + lap_z

        return out

    return laplace  # type: ignore


@CartesianGridBase.register_operator("laplace", rank_in=0, rank_out=0)
def make_laplace(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a laplace operator on a Cartesian grid

    Args:
    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str): Method used for calculating the laplace operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim

    if method == "auto":
        # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
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

    # elif method == "matrix":
    #     laplace = make_laplace_from_matrix(*_get_laplace_matrix(grid))

    elif method == "scipy":
        laplace = _make_laplace_scipy_nd(grid)

    else:
        raise ValueError(f"Method `{method}` is not defined")

    return laplace


def _make_gradient_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a gradient operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = 0.5 / grid.discretization
    dim = grid.dim
    shape_out = (dim,) + grid.shape

    def gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        assert arr.shape == tuple(s + 2 for s in grid.shape)
        if out is None:
            out = np.empty(shape_out)
        else:
            assert out.shape == shape_out

        for i in range(dim):
            out[i] = ndimage.convolve1d(arr, [1, 0, -1], axis=i) * scaling[i]
        return out

    return gradient


def _make_gradient_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = 0.5 / grid.discretization[0]

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[0, i] = (arr[i + 1] - arr[i - 1]) * scale

        return out

    return gradient  # type: ignore


def _make_gradient_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                out[0, i, j] = (arr[i + 1, j] - arr[i - 1, j]) * scale_x
                out[1, i, j] = (arr[i, j + 1] - arr[i, j - 1]) * scale_y

        return out

    return gradient  # type: ignore


def _make_gradient_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    out[0, i, j] = (arr[i + 1, j, k] - arr[i - 1, j, k]) * scale_x
                    out[1, i, j] = (arr[i, j + 1, k] - arr[i, j - 1, k]) * scale_y
                    out[2, i, j] = (arr[i, j, k + 1] - arr[i, j, k - 1]) * scale_z

        return out

    return gradient  # type: ignore


@CartesianGridBase.register_operator("gradient", rank_in=0, rank_out=1)
def make_gradient(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a gradient operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str): Method used for calculating the gradient operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    grid.check_value_rank(0)

    if method == "auto":
        # choose the fastest available gradient operator
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
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

    elif method == "scipy":
        gradient = _make_gradient_scipy_nd(grid)

    else:
        raise ValueError(f"Method `{method}` is not defined")

    return gradient


def _make_gradient_squared_numba_1d(
    grid: CartesianGridBase, central: bool = True
) -> OperatorType:
    """make a 1d squared gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
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
        scale = 1 / (2 * grid.discretization[0]) ** 2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                out[i] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

            return out

    else:
        # use forward and backward differences
        scale = 1 / (2 * grid.discretization[0] ** 2)

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_x + 1):
                diff_l = (arr[i + 1] - arr[i]) ** 2
                diff_r = (arr[i] - arr[i - 1]) ** 2
                out[i] = (diff_l + diff_r) * scale

            return out

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_2d(
    grid: CartesianGridBase, central: bool = True
) -> OperatorType:
    """make a 2d squared gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
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
    parallel = dim_x * dim_y >= config["numba.parallel_threshold"]

    if central:
        # use central differences
        scale_x, scale_y = 1 / (2 * grid.discretization) ** 2

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
            """apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    term_x = (arr[i + 1, j] - arr[i - 1, j]) ** 2 * scale_x
                    term_y = (arr[i, j] - arr[i, j - 1]) ** 2 * scale_y
                    out[i, j] = term_x + term_y

            return out

    else:
        # use forward and backward differences
        scale_x, scale_y = 1 / (2 * grid.discretization ** 2)

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
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
                    out[i, j] = term_x + term_y

            return out

    return gradient_squared  # type: ignore


def _make_gradient_squared_numba_3d(
    grid: CartesianGridBase, central: bool = True
) -> OperatorType:
    """make a 3d squared gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
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
    parallel = dim_x * dim_y * dim_z >= config["numba.parallel_threshold"]

    if central:
        # use central differences
        scale_x, scale_y, scale_z = 1 / (2 * grid.discretization) ** 2

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
            """apply squared gradient operator to array `arr`"""
            for i in nb.prange(1, dim_x + 1):
                for j in range(1, dim_y + 1):
                    for k in range(1, dim_z + 1):
                        term_x = (arr[i + 1, j, k] - arr[i - 1, j, k]) ** 2 * scale_x
                        term_y = (arr[i, j + 1, k] - arr[i, j - 1, k]) ** 2 * scale_y
                        term_z = (arr[i, j, k + 1] - arr[i, j, k - 1]) ** 2 * scale_z
                        out[i, j, k] = term_x + term_y + term_z

            return out

    else:
        # use forward and backward differences
        scale_x, scale_y, scale_z = 1 / (2 * grid.discretization ** 2)

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
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
                            + (arr[i, j, k] - [i, j, k - 1]) ** 2
                        ) * scale_z
                        out[i, j, k] = term_x + term_y + term_z

            return out

    return gradient_squared  # type: ignore


@CartesianGridBase.register_operator("gradient_squared", rank_in=0, rank_out=0)
def make_gradient_squared(
    grid: CartesianGridBase, central: bool = True
) -> OperatorType:
    """make a gradient operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
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
    grid.check_value_rank(0)

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


def _make_divergence_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a divergence operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    shape = grid.shape
    scaling = 0.5 / grid.discretization

    def divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply divergence operator to array `arr`"""
        data_full_shape = tuple(s + 2 for s in grid.shape)
        assert arr.shape[0] == len(shape) and arr.shape[1:] == data_full_shape

        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(arr.shape[1:])
        else:
            out[:] = 0

        for i in range(len(shape)):
            out += ndimage.convolve1d(arr[i], [1, 0, -1], axis=i) * scaling[i]
        return out

    return divergence


def _make_divergence_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x = grid.shape[0]
    scale = 0.5 / grid.discretization[0]

    @jit
    def divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_x + 1):
            out[i] = (arr[0, i + 1] - arr[0, i - 1]) * scale

        return out

    return divergence  # type: ignore


def _make_divergence_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y = grid.shape
    scale_x, scale_y = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                d_x = (arr[0, i + 1, j] - arr[0, i - 1, j]) * scale_x
                d_y = (arr[1, i, j + 1] - arr[1, i, j - 1]) * scale_y
                out[i, j] = d_x + d_y

        return out

    return divergence  # type: ignore


def _make_divergence_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d divergence operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = grid.shape
    scale_x, scale_y, scale_z = 0.5 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y * dim_z >= config["numba.parallel_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_x + 1):
            for j in range(1, dim_y + 1):
                for k in range(1, dim_z + 1):
                    d_x = (arr[0, i + 1, j, k] - arr[0, i - 1, j, k]) * scale_x
                    d_y = (arr[1, i, j + 1, k] - arr[1, i, j - 1, k]) * scale_y
                    d_z = (arr[1, i, j, k + 1] - arr[1, i, j, k - 1]) * scale_z
                    out[i, j, k] = d_x + d_y + d_z

        return out

    return divergence  # type: ignore


@CartesianGridBase.register_operator("divergence", rank_in=1, rank_out=0)
def make_divergence(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a divergence operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str): Method used for calculating the divergence operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    grid.check_value_rank(0)

    if method == "auto":
        # choose the fastest available divergence operator
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
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

    elif method == "scipy":
        divergence = _make_divergence_scipy_nd(grid)

    else:
        raise ValueError(f"Method `{method}` is not defined")

    return divergence


def _make_vector_gradient_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a vector gradient operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = 0.5 / grid.discretization
    dim = grid.dim
    shape_out = (dim, dim) + tuple(s + 2 for s in grid.shape)

    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply vector gradient operator to array `arr`"""
        assert arr.shape == shape_out[1:]
        if out is None:
            out = np.empty(shape_out)
        else:
            assert out.shape == shape_out

        for i in range(dim):
            for j in range(dim):
                conv = ndimage.convolve1d(arr[j], [1, 0, -1], axis=i)
                out[i, j] = conv * scaling[i]
        return out

    return vector_gradient


def _make_vector_gradient_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d vector gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    gradient = _make_gradient_numba_1d(grid)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        gradient(arr[0], out=out[0])
        return out

    return vector_gradient  # type: ignore


def _make_vector_gradient_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d vector gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    gradient = _make_gradient_numba_2d(grid)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        gradient(arr[0], out=out[:, 0])
        gradient(arr[1], out=out[:, 1])
        return out

    return vector_gradient  # type: ignore


def _make_vector_gradient_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d vector gradient operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    gradient = _make_gradient_numba_3d(grid)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        gradient(arr[0], out=out[:, 0])
        gradient(arr[1], out=out[:, 1])
        gradient(arr[2], out=out[:, 2])
        return out

    return vector_gradient  # type: ignore


@CartesianGridBase.register_operator("vector_gradient", rank_in=1, rank_out=2)
def make_vector_gradient(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a vector gradient operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str): Method used for calculating the vector gradient operator.
            If method='auto', a suitable method is chosen automatically

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    grid.check_value_rank(1)

    # choose the fastest available vector gradient operator
    if method == "auto":
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
        if dim == 1:
            gradient = _make_vector_gradient_numba_1d(grid)
        elif dim == 2:
            gradient = _make_vector_gradient_numba_2d(grid)
        elif dim == 3:
            gradient = _make_vector_gradient_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba vector gradient operator not implemented for dimension {dim}"
            )

    elif method == "scipy":
        gradient = _make_vector_gradient_scipy_nd(grid)
    else:
        raise ValueError(f"Method `{method}` is not defined")

    return gradient


def _make_vector_laplace_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a vector Laplacian using the scipy module

    This only supports uniform discretizations.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = grid._uniform_discretization ** -2
    dim = grid.dim
    shape_out = (dim,) + tuple(s + 2 for s in grid.shape)

    def vector_laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply vector Laplacian operator to array `arr`"""
        assert arr.shape == shape_out
        if out is None:
            out = np.empty(shape_out)
        else:
            assert out.shape == shape_out

        for i in range(dim):
            ndimage.laplace(arr[i], output=out[i])
        return out * scaling

    return vector_laplace


def _make_vector_laplace_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d vector Laplacian using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    laplace = _make_laplace_numba_1d(grid)

    @jit
    def vector_laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply vector Laplacian to array `arr`"""
        laplace(arr[0], out=out[0])
        return out

    return vector_laplace  # type: ignore


def _make_vector_laplace_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d vector Laplacian using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    laplace = _make_laplace_numba_2d(grid)

    @jit
    def vector_laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply vector Laplacian  to array `arr`"""
        laplace(arr[0], out=out[0])
        laplace(arr[1], out=out[1])
        return out

    return vector_laplace  # type: ignore


def _make_vector_laplace_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d vector Laplacian using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    laplace = _make_laplace_numba_3d(grid)

    @jit
    def vector_laplace(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply vector Laplacian to array `arr`"""
        laplace(arr[0], out=out[0])
        laplace(arr[1], out=out[1])
        laplace(arr[2], out=out[2])
        return out

    return vector_laplace  # type: ignore


@CartesianGridBase.register_operator("vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a vector Laplacian on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str):
            Method used for calculating the vector laplace operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    grid.check_value_rank(1)

    # choose the fastest available vector gradient operator
    if method == "auto":
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
        if dim == 1:
            gradient = _make_vector_laplace_numba_1d(grid)
        elif dim == 2:
            gradient = _make_vector_laplace_numba_2d(grid)
        elif dim == 3:
            gradient = _make_vector_laplace_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba vector gradient operator not implemented for dimension {dim}"
            )

    elif method == "scipy":
        gradient = _make_vector_laplace_scipy_nd(grid)
    else:
        raise ValueError(f"Method `{method}` is not defined")

    return gradient


def _make_tensor_divergence_scipy_nd(grid: CartesianGridBase) -> OperatorType:
    """make a tensor divergence operator using the scipy module

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    from scipy import ndimage

    scaling = 0.5 / grid.discretization
    dim = grid.dim
    shape_out = (dim,) + tuple(s + 2 for s in grid.shape)

    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply tensor divergence operator to array `arr`"""
        # need to initialize with zeros since data is added later
        assert arr.shape[0] == dim and arr.shape[1:] == shape_out
        if out is None:
            out = np.zeros(shape_out)
        else:
            assert out.shape == shape_out
            out[:] = 0

        for i in range(dim):
            for j in range(dim):
                conv = ndimage.convolve1d(arr[i, j], [1, 0, -1], axis=j)
                out[i] += conv * scaling[j]
        return out

    return tensor_divergence


def _make_tensor_divergence_numba_1d(grid: CartesianGridBase) -> OperatorType:
    """make a 1d tensor divergence  operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    divergence = _make_divergence_numba_1d(grid)

    @jit
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        divergence(arr[0], out=out[0])
        return out

    return tensor_divergence  # type: ignore


def _make_tensor_divergence_numba_2d(grid: CartesianGridBase) -> OperatorType:
    """make a 2d tensor divergence  operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    divergence_x = _make_divergence_numba_2d(grid)
    divergence_y = _make_divergence_numba_2d(grid)

    @jit
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        divergence_x(arr[0], out=out[0])
        divergence_y(arr[1], out=out[1])
        return out

    return tensor_divergence  # type: ignore


def _make_tensor_divergence_numba_3d(grid: CartesianGridBase) -> OperatorType:
    """make a 3d tensor divergence  operator using numba compilation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    divergence_x = _make_divergence_numba_3d(grid)
    divergence_y = _make_divergence_numba_3d(grid)
    divergence_z = _make_divergence_numba_3d(grid)

    @jit
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> np.ndarray:
        """apply gradient operator to array `arr`"""
        divergence_x(arr[0], out=out[0])
        divergence_y(arr[1], out=out[1])
        divergence_z(arr[2], out=out[2])
        return out

    return tensor_divergence  # type: ignore


@CartesianGridBase.register_operator("tensor_divergence", rank_in=2, rank_out=1)
def make_tensor_divergence(
    grid: CartesianGridBase, method: str = "auto"
) -> OperatorType:
    """make a tensor divergence operator on a Cartesian grid

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str):
            Method used for calculating the tensor divergence operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    dim = grid.dim
    grid.check_value_rank(1)

    # choose the fastest available tensor divergence operator
    if method == "auto":
        if 1 <= dim <= 3:
            method = "numba"
        else:
            method = "scipy"

    if method == "numba":
        if dim == 1:
            func = _make_tensor_divergence_numba_1d(grid)
        elif dim == 2:
            func = _make_tensor_divergence_numba_2d(grid)
        elif dim == 3:
            func = _make_tensor_divergence_numba_3d(grid)
        else:
            raise NotImplementedError(
                f"Numba tensor divergence operator not implemented for dimension {dim}"
            )

    elif method == "scipy":
        func = _make_tensor_divergence_scipy_nd(grid)
    else:
        raise ValueError(f"Method `{method}` is not defined")

    return func


@CartesianGridBase.register_operator("poisson_solver", rank_in=0, rank_out=0)
def make_poisson_solver(grid: CartesianGridBase, method: str = "auto") -> OperatorType:
    """make a operator that solves Poisson's equation

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGridBase`):
            The grid for which the operator is created
        method (str):
            Method used for calculating the tensor divergence operator.
            If method='auto', a suitable method is chosen automatically.

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(grid)
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
