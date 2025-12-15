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

from ....grids.spherical import PolarSymGrid
from ....tools.docstrings import fill_in_docstring
from .. import numba_backend
from ..utils import jit

if TYPE_CHECKING:
    from ....tools.typing import NumericArray, OperatorImplType


@numba_backend.register_operator(PolarSymGrid, "laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized laplace operator for a polar grid.

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    factor_r = 1 / (2 * grid.axes_coords[0] * dr)
    dr_2 = 1 / dr**2

    @jit
    def laplace(arr: NumericArray, out: NumericArray) -> None:
        """Apply laplace operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr_2
            out[i - 1] += (arr[i + 1] - arr[i - 1]) * factor_r[i - 1]

    return laplace  # type: ignore


@numba_backend.register_operator(PolarSymGrid, "gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(
    grid: PolarSymGrid, *, method: Literal["central", "forward", "backward"] = "central"
) -> OperatorImplType:
    """Make a discretized gradient operator for a polar grid.

    {DESCR_POLAR_GRID}

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
    dim_r = grid.shape[0]
    if method == "central":
        scale_r = 0.5 / grid.discretization[0]
    elif method in {"forward", "backward"}:
        scale_r = 1 / grid.discretization[0]
    else:
        msg = f"Unknown derivative type `{method}`"
        raise ValueError(msg)

    @jit
    def gradient(arr: NumericArray, out: NumericArray) -> None:
        """Apply gradient operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            if method == "central":
                out[0, i - 1] = (arr[i + 1] - arr[i - 1]) * scale_r
            elif method == "forward":
                out[0, i - 1] = (arr[i + 1] - arr[i]) * scale_r
            elif method == "backward":
                out[0, i - 1] = (arr[i] - arr[i - 1]) * scale_r
            out[1, i - 1] = 0  # no angular dependence by definition

    return gradient  # type: ignore


@numba_backend.register_operator(
    PolarSymGrid, "gradient_squared", rank_in=0, rank_out=0
)
@fill_in_docstring
def make_gradient_squared(
    grid: PolarSymGrid, *, central: bool = True
) -> OperatorImplType:
    """Make a discretized gradient squared operator for a polar grid.

    {DESCR_POLAR_GRID}

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
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    if central:
        # use central differences
        scale = 0.25 / dr**2

        @jit
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / dr**2

        @jit
        def gradient_squared(arr: NumericArray, out: NumericArray) -> None:
            """Apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                term = (arr[i + 1] - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = term * scale

    return gradient_squared  # type: ignore


@numba_backend.register_operator(PolarSymGrid, "divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized divergence operator for a polar grid.

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    scale_r = 1 / (2 * dr)

    @jit
    def divergence(arr: NumericArray, out: NumericArray) -> None:
        """Apply divergence operator to array `arr`"""
        # inner radial boundary condition
        for i in range(1, dim_r + 1):  # iterate radial points
            out[i - 1] = (arr[0, i + 1] - arr[0, i - 1]) * scale_r
            out[i - 1] += arr[0, i] / rs[i - 1]

    return divergence  # type: ignore


@numba_backend.register_operator(PolarSymGrid, "vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized vector gradient operator for a polar grid.

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def vector_gradient(arr: NumericArray, out: NumericArray) -> None:
        """Apply vector gradient operator to array `arr`"""
        # assign aliases
        arr_r, arr_φ = arr
        out_rr, out_rφ = out[0, 0, :], out[0, 1, :]
        out_φr, out_φφ = out[1, 0, :], out[1, 1, :]

        for i in range(1, dim_r + 1):  # iterate radial points
            out_rr[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
            out_rφ[i - 1] = -arr_φ[i] / rs[i - 1]
            out_φr[i - 1] = (arr_φ[i + 1] - arr_φ[i - 1]) * scale_r
            out_φφ[i - 1] = arr_r[i] / rs[i - 1]

    return vector_gradient  # type: ignore


@numba_backend.register_operator(
    PolarSymGrid, "tensor_divergence", rank_in=2, rank_out=1
)
@fill_in_docstring
def make_tensor_divergence(grid: PolarSymGrid) -> OperatorImplType:
    """Make a discretized tensor divergence operator for a polar grid.

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def tensor_divergence(arr: NumericArray, out: NumericArray) -> None:
        """Apply tensor divergence operator to array `arr`"""
        # assign aliases
        arr_rr, arr_rφ = arr[0, 0, :], arr[0, 1, :]
        arr_φr, arr_φφ = arr[1, 0, :], arr[1, 1, :]
        out_r, out_φ = out[0, :], out[1, :]

        # iterate over inner points
        for i in range(1, dim_r + 1):
            term = (arr_rr[i] - arr_φφ[i]) / rs[i - 1]
            out_r[i - 1] = (arr_rr[i + 1] - arr_rr[i - 1]) * scale_r + term
            term = (arr_rφ[i] + arr_φr[i]) / rs[i - 1]
            out_φ[i - 1] = (arr_φr[i + 1] - arr_φr[i - 1]) * scale_r + term

    return tensor_divergence  # type: ignore
