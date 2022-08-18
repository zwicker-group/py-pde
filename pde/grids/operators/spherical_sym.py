r"""
This module implements differential operators on spherical grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_tensor_divergence
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Tuple

import numpy as np

from ...tools.docstrings import fill_in_docstring
from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..boundaries import Boundaries
from ..spherical import SphericalSymGrid
from .common import make_general_poisson_solver


@SphericalSymGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(grid: SphericalSymGrid, conservative: bool = True) -> OperatorType:
    """make a discretized laplace operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the laplace operator should be conservative (which
            results in slightly slower computations). Conservative operators ensure mass
            conservation.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    r_min, r_max = grid.axes_bounds[0]

    if conservative:
        # create a conservative spherical laplace operator
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        assert np.isclose(rl[0], r_min) and np.isclose(rh[-1], r_max)
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (dr * volumes)
        factor_h = rh**2 / (dr * volumes)

        @jit
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
            """apply laplace operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                term_h = factor_h[i - 1] * (arr[i + 1] - arr[i])
                term_l = factor_l[i - 1] * (arr[i] - arr[i - 1])
                out[i - 1] = term_h - term_l

    else:  # create an operator that is not conservative
        dr2 = 1 / dr**2

        @jit
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
            """apply laplace operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                diff_2 = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr2
                diff_1 = (arr[i + 1] - arr[i - 1]) / (rs[i - 1] * dr)
                out[i - 1] = diff_2 + diff_1

    return laplace  # type: ignore


@SphericalSymGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(grid: SphericalSymGrid) -> OperatorType:
    """make a discretized gradient operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    scale_r = 1 / (2 * dr)

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            out[0, i - 1] = (arr[i + 1] - arr[i - 1]) * scale_r
            out[1, i - 1] = out[2, i - 1] = 0  # no angular dependence by definition

    return gradient  # type: ignore


@SphericalSymGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(grid: SphericalSymGrid, central: bool = True) -> OperatorType:
    """make a discretized gradient squared operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
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
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    if central:
        # use central differences
        scale = 0.25 / dr**2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / dr**2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                term = (arr[i + 1] - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = term * scale

    return gradient_squared  # type: ignore


@SphericalSymGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(
    grid: SphericalSymGrid, safe: bool = True, conservative: bool = True
) -> OperatorType:
    """make a discretized divergence operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Warning:
        This operator ignores the θ-component of the field when calculating the
        divergence. This is because the resulting scalar field could not be expressed
        on a :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]

    if conservative:
        # implement conservative version of the divergence operator
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (2 * volumes)
        factor_h = rh**2 / (2 * volumes)

        @jit
        def divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply divergence operator to array `arr`"""
            if safe:
                # the θ-component of the vector field are required to be zero. If this
                # was not the case the scale field resulting from the divergence would
                # contain components that cannot be expressed in spherically symmetric
                # coordinates
                assert np.all(arr[1, 1:-1] == 0)

            arr_r = arr[0, :]
            for i in range(1, dim_r + 1):  # iterate radial points
                term_h = factor_h[i - 1] * (arr_r[i] + arr_r[i + 1])
                term_l = factor_l[i - 1] * (arr_r[i - 1] + arr_r[i])
                out[i - 1] = term_h - term_l

    else:
        # implement naive divergence operator
        scale_r = 1 / (2 * dr)
        factors = 2 / rs  # factors that need to be multiplied below

        @jit
        def divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply divergence operator to array `arr`"""
            if safe:
                # the θ-component of the vector field are required to be zero. If this
                # was not the case the scale field resulting from the divergence would
                # contain components that cannot be expressed in spherically symmetric
                # coordinates
                assert np.all(arr[1, 1:-1] == 0)

            arr_r = arr[0, :]
            for i in range(1, dim_r + 1):  # iterate radial points
                diff_r = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
                out[i - 1] = diff_r + factors[i - 1] * arr_r[i]

    return divergence  # type: ignore


@SphericalSymGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(grid: SphericalSymGrid, safe: bool = True) -> OperatorType:
    """make a discretized vector gradient operator for a spherical grid

    Warning:
        This operator ignores the two angular components of the field when calculating
        the gradient. This is because the resulting field could not be expressed on a
        :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply vector gradient operator to array `arr`"""
        if safe:
            # the θ- and φ-components are required to be zero. If this was not the case
            # the tensor field resulting from the gradient would contain components that
            # cannot be expressed in spherically symmetric coordinates
            assert np.all(arr[1:, 1:-1] == 0)

        # assign aliases
        arr_r = arr[0, :]
        out_rr, out_rθ, out_rφ = out[0, 0, :], out[0, 1, :], out[0, 2, :]
        out_θr, out_θθ, out_θφ = out[1, 0, :], out[1, 1, :], out[1, 2, :]
        out_φr, out_φθ, out_φφ = out[2, 0, :], out[2, 1, :], out[2, 2, :]

        # set all components to zero that are not affected
        out_rθ[:] = 0
        out_rφ[:] = 0
        out_θr[:] = 0
        out_θφ[:] = 0
        out_φr[:] = 0
        out_φθ[:] = 0

        # inner radial boundary condition
        for i in range(1, dim_r + 1):  # iterate radial points
            out_rr[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
            out_θθ[i - 1] = arr_r[i] / rs[i - 1]
            out_φφ[i - 1] = arr_r[i] / rs[i - 1]

    return vector_gradient  # type: ignore


@SphericalSymGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(
    grid: SphericalSymGrid, safe: bool = True, conservative: bool = True
) -> OperatorType:
    """make a discretized tensor divergence operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]

    if conservative:
        # conservative implementation of the tensor divergence
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl**2 / (2 * volumes)
        factor_h = rh**2 / (2 * volumes)

        @jit
        def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply tensor divergence operator to array `arr`"""
            # assign aliases
            arr_rr, arr_rθ, arr_rφ = arr[0, 0, :], arr[0, 1, :], arr[0, 2, :]
            arr_θr, arr_θθ, arr_θφ = arr[1, 0, :], arr[1, 1, :], arr[1, 2, :]
            arr_φr, arr_φθ, arr_φφ = arr[2, 0, :], arr[2, 1, :], arr[2, 2, :]
            out_r, out_θ, out_φ = out[0, :], out[1, :], out[2, :]

            # check inputs
            if safe:
                # the following conditions need to be met. Otherwise, the vector resulting
                # from the divergence might contain components that cannot be expressed in
                # spherically symmetric coordinates
                assert np.all(arr_φr[1:-1] == 0)
                assert np.all(arr_rφ[1:-1] == 0)
                assert np.all(arr_rθ[1:-1] == 0)
                assert np.all(arr_θr[1:-1] == 0)
                assert np.all(arr_θθ[1:-1] == arr_φφ[1:-1])
                assert np.all(arr_φθ[1:-1] == -arr_θφ[1:-1])

            # iterate over inner points
            for i in range(1, dim_r + 1):
                term_r_h = factor_h[i - 1] * (arr_rr[i] + arr_rr[i + 1])
                term_r_l = factor_l[i - 1] * (arr_rr[i - 1] + arr_rr[i])
                out_r[i - 1] = term_r_h - term_r_l - 2 * arr_φφ[i] / rs[i - 1]
                out_θ[i - 1] = 0
                out_φ[i - 1] = 0

    else:
        # naive implementation of the tensor divergence
        scale_r = 1 / (2 * dr)

        @jit
        def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply tensor divergence operator to array `arr`"""
            # assign aliases
            arr_rr, arr_rθ, arr_rφ = arr[0, 0, :], arr[0, 1, :], arr[0, 2, :]
            arr_θr, arr_θθ, arr_θφ = arr[1, 0, :], arr[1, 1, :], arr[1, 2, :]
            arr_φr, arr_φθ, arr_φφ = arr[2, 0, :], arr[2, 1, :], arr[2, 2, :]
            out_r, out_θ, out_φ = out[0, :], out[1, :], out[2, :]

            # check inputs
            if safe:
                # the following conditions need to be met. Otherwise, the vector resulting
                # from the divergence might contain components that cannot be expressed in
                # spherically symmetric coordinates
                assert np.all(arr_rθ[1:-1] == 0)
                assert np.all(arr_θθ[1:-1] == arr_φφ[1:-1])
                assert np.all(arr_φθ[1:-1] == -arr_θφ[1:-1])

            # iterate over inner points
            for i in range(1, dim_r + 1):
                deriv_r = (arr_rr[i + 1] - arr_rr[i - 1]) * scale_r
                out_r[i - 1] = deriv_r + 2 * (arr_rr[i] - arr_φφ[i]) / rs[i - 1]

                deriv_r = (arr_θr[i + 1] - arr_θr[i - 1]) * scale_r
                out_θ[i - 1] = deriv_r + 2 * arr_θr[i] / rs[i - 1]

                deriv_r = (arr_φr[i + 1] - arr_φr[i - 1]) * scale_r
                out_φ[i - 1] = deriv_r + (2 * arr_φr[i] + arr_rφ[i]) / rs[i - 1]

    return tensor_divergence  # type: ignore


@SphericalSymGrid.register_operator("tensor_double_divergence", rank_in=2, rank_out=0)
@fill_in_docstring
def make_tensor_double_divergence(
    grid: SphericalSymGrid, safe: bool = True, conservative: bool = True
) -> OperatorType:
    """make a discretized tensor double divergence operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input
        conservative (bool):
            Flag indicating whether the operator should be conservative (which results
            in slightly slower computations). Conservative operators ensure mass
            conservation.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]

    if conservative:
        # create a conservative double divergence laplace operator
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        r_min, r_max = grid.axes_bounds[0]
        assert np.isclose(rl[0], r_min) and np.isclose(rh[-1], r_max)
        volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells
        factor_l = rl / volumes
        factor_h = rh / volumes
        factor2_l = rl**2 / (dr * volumes)
        factor2_h = rh**2 / (dr * volumes)

        @jit
        def tensor_double_divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply double divergence operator to tensor array `arr`"""
            # assign aliases
            arr_rr, arr_rθ, ______ = arr[0, 0, :], arr[0, 1, :], arr[0, 2, :]
            arr_θr, arr_θθ, ______ = arr[1, 0, :], arr[1, 1, :], arr[1, 2, :]
            ______, ______, arr_φφ = arr[2, 0, :], arr[2, 1, :], arr[2, 2, :]

            # check inputs
            if safe:
                # the following conditions need to be met. Otherwise, the vector
                # resulting from the divergence might contain components that cannot be
                # expressed in spherically symmetric coordinates
                assert np.all(arr_rθ[1:-1] == -arr_θr[1:-1])
                assert np.all(arr_θθ[1:-1] == arr_φφ[1:-1])

            # iterate over inner points
            for i in range(1, dim_r + 1):
                # radial part
                arr_rr_h = arr_rr[i] + arr_rr[i + 1]
                arr_rr_l = arr_rr[i - 1] + arr_rr[i]
                arr_rr_dr_h = arr_rr[i + 1] - arr_rr[i]
                arr_rr_dr_l = arr_rr[i] - arr_rr[i - 1]
                div2_rr_h = factor_h[i - 1] * arr_rr_h + factor2_h[i - 1] * arr_rr_dr_h
                div2_rr_l = factor_l[i - 1] * arr_rr_l + factor2_l[i - 1] * arr_rr_dr_l
                div2_rr = div2_rr_h - div2_rr_l

                # angular part
                arr_φφ_h = arr_φφ[i] + arr_φφ[i + 1]
                arr_φφ_l = arr_φφ[i - 1] + arr_φφ[i]
                div2_φφ = factor_h[i - 1] * arr_φφ_h - factor_l[i - 1] * arr_φφ_l

                out[i - 1] = div2_rr - div2_φφ

    else:
        # naive, non-conservative implementation of the double divergence
        dr2 = 1 / dr**2
        scale_r = 1 / (2 * dr)

        @jit
        def tensor_double_divergence(arr: np.ndarray, out: np.ndarray) -> None:
            """apply double divergence operator to tensor array `arr`"""
            # assign aliases
            arr_rr, arr_rθ, ______ = arr[0, 0, :], arr[0, 1, :], arr[0, 2, :]
            arr_θr, arr_θθ, ______ = arr[1, 0, :], arr[1, 1, :], arr[1, 2, :]
            ______, ______, arr_φφ = arr[2, 0, :], arr[2, 1, :], arr[2, 2, :]

            # check inputs
            if safe:
                # the following conditions need to be met. Otherwise, the vector
                # resulting from the divergence might contain components that cannot be
                # expressed in spherically symmetric coordinates
                assert np.all(arr_rθ[1:-1] == -arr_θr[1:-1])
                assert np.all(arr_θθ[1:-1] == arr_φφ[1:-1])

            # iterate over inner points
            for i in range(1, dim_r + 1):
                # first derivatives
                arr_rr_dr = (arr_rr[i + 1] - arr_rr[i - 1]) * scale_r
                arr_φφ_dr = (arr_φφ[i + 1] - arr_φφ[i - 1]) * scale_r

                # second derivative
                term1 = (arr_rr[i + 1] - arr_rr[i - 1]) / (rs[i - 1] * dr)
                term2 = (arr_rr[i + 1] - 2 * arr_rr[i] + arr_rr[i - 1]) * dr2
                lap_rr = term1 + term2

                enum = (arr_rr[i] - arr_φφ[i]) / rs[i - 1] + arr_rr_dr - arr_φφ_dr
                out[i - 1] = lap_rr + 2 * enum / rs[i - 1]

    return tensor_double_divergence  # type: ignore


@fill_in_docstring
def _get_laplace_matrix(bcs: Boundaries) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a polar grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    assert isinstance(bcs.grid, SphericalSymGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, r_max = bcs.grid.axes_bounds[0]

    # create a conservative spherical laplace operator
    rl = r_min + dr * np.arange(dim_r)  # inner radii of spherical shells
    rh = rl + dr  # outer radii
    assert np.isclose(rh[-1], r_max)
    volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells

    factor_l = (rs - 0.5 * dr) ** 2 / (dr * volumes)
    factor_h = (rs + 0.5 * dr) ** 2 / (dr * volumes)

    matrix = sparse.dok_matrix((dim_r, dim_r))
    vector = sparse.dok_matrix((dim_r, 1))

    for i in range(dim_r):
        matrix[i, i] += -factor_l[i] - factor_h[i]

        if i == 0:
            if r_min == 0:
                matrix[i, i + 1] = factor_l[i]
            else:
                const, entries = bcs[0].get_data((-1,))
                vector[i] += const * factor_l[i]
                for k, v in entries.items():
                    matrix[i, k] += v * factor_l[i]

        else:
            matrix[i, i - 1] = factor_l[i]

        if i == dim_r - 1:
            const, entries = bcs[0].get_data((dim_r,))
            vector[i] += const * factor_h[i]
            for k, v in entries.items():
                matrix[i, k] += v * factor_h[i]

        else:
            matrix[i, i + 1] = factor_h[i]

    return matrix, vector


@SphericalSymGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
@fill_in_docstring
def make_poisson_solver(bcs: Boundaries, method: str = "auto") -> OperatorType:
    """make a operator that solves Poisson's equation

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str):
            The chosen method for implementing the operator

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)
