"""This module implements differential operators on Cartesian grids using scipy.

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

from typing import TYPE_CHECKING, Literal

import numpy as np

from ....grids.cartesian import CartesianGrid
from .. import scipy_backend
from .common import make_general_poisson_solver, uniform_discretization

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....grids.boundaries.axes import BoundariesList
    from ....tools.typing import NumericArray, OperatorImplType


def _get_laplace_matrix_1d(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
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


def _get_laplace_matrix_2d(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
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


def _get_laplace_matrix_3d(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
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


def _get_laplace_matrix(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
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
        msg = f"{dim:d}-dimensional Laplace matrix not implemented"
        raise NotImplementedError(msg)

    return result


@scipy_backend.register_operator(CartesianGrid, "laplace", rank_in=0, rank_out=0)
def make_laplace(grid: CartesianGrid, **kwargs) -> OperatorImplType:
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

    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply Laplace operator to array `arr`"""
        assert arr.shape == grid._shape_full
        valid = (...,) + (slice(1, -1),) * grid.dim
        with np.errstate(all="ignore"):
            # some errors can happen for ghost cells
            out[:] = ndimage.laplace(scaling * arr)[valid]

    return laplace


@scipy_backend.register_operator(CartesianGrid, "gradient", rank_in=0, rank_out=1)
def make_gradient(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
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
    shape_out = (dim, *grid.shape)

    if method == "central":
        stencil = [-0.5, 0, 0.5]
    elif method == "forward":
        stencil = [0, -1, 1]
    elif method == "backward":
        stencil = [-1, 1, 0]
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def gradient(arr: NumericArray, out: NumericArray) -> None:
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


@scipy_backend.register_operator(CartesianGrid, "divergence", rank_in=1, rank_out=0)
def make_divergence(
    grid: CartesianGrid,
    *,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorImplType:
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
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    def divergence(arr: NumericArray, out: NumericArray) -> None:
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

    return vectorized_operator


@scipy_backend.register_operator(
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
    return _vectorize_operator(make_gradient, grid, method=method)


@scipy_backend.register_operator(CartesianGrid, "vector_laplace", rank_in=1, rank_out=1)
def make_vector_laplace(grid: CartesianGrid) -> OperatorImplType:
    """Make a vector Laplacian on a Cartesian grid.

    Args:
        grid (:class:`~pde.grids.cartesian.CartesianGrid`):
            The grid for which the operator is created
        backend (str):
            Backend used for calculating the vector Laplace operator.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_laplace, grid)


@scipy_backend.register_operator(
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
        backend (str):
            Backend used for calculating the tensor divergence operator.
        method (str):
            The method for calculating the derivative. Possible values are 'central',
            'forward', and 'backward'.

    Returns:
        A function that can be applied to an array of values
    """
    return _vectorize_operator(make_divergence, grid, method=method)


@scipy_backend.register_operator(CartesianGrid, "poisson_solver", rank_in=0, rank_out=0)
def make_poisson_solver(
    bcs: BoundariesList, *, method: Literal["auto", "scipy"] = "auto"
) -> OperatorImplType:
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
    "make_divergence",
    "make_gradient",
    "make_laplace",
    "make_poisson_solver",
    "make_tensor_divergence",
    "make_vector_gradient",
    "make_vector_laplace",
]
