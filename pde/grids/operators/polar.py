r"""
This module implements differential operators on polar grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_tensor_divergence
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable

from scipy import sparse

from ...tools.docstrings import fill_in_docstring
from ...tools.numba import jit_allocate_out
from ..boundaries import Boundaries
from ..spherical import PolarGrid
from .common import make_general_poisson_solver


@PolarGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(bcs: Boundaries) -> Callable:
    """make a discretized laplace operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    dr_2 = 1 / dr ** 2

    # prepare boundary values
    value_lower_bc = bcs[0].low.make_virtual_point_evaluator()
    value_upper_bc = bcs[0].high.make_virtual_point_evaluator()

    @jit_allocate_out(out_shape=(dim_r,))
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        i = 0
        if r_min == 0:
            # Apply Neumann condition at the origin
            out[i] = 2 * (arr[i + 1] - arr[i]) * dr_2
        else:
            arr_r_l = value_lower_bc(arr, (i,))
            out[i] = (arr[i + 1] - 2 * arr[i] + arr_r_l) * dr_2
            out[i] += (arr[i + 1] - arr_r_l) / (2 * rs[i] * dr)

        for i in range(1, dim_r - 1):  # iterate inner radial points
            out[i] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr_2
            out[i] += (arr[i + 1] - arr[i - 1]) / (2 * rs[i] * dr)

        # express boundary condition at outer side
        i = dim_r - 1
        arr_r_h = value_upper_bc(arr, (i,))
        out[i] = (arr_r_h - 2 * arr[i] + arr[i - 1]) * dr_2
        out[i] += (arr_r_h - arr[i - 1]) / (2 * rs[i] * dr)
        return out

    return laplace  # type: ignore


@PolarGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(bcs: Boundaries) -> Callable:
    """make a discretized gradient operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    dr = bcs.grid.discretization[0]
    scale_r = 1 / (2 * dr)

    # prepare boundary values
    boundary = bcs[0]
    value_lower_bc = boundary.low.make_virtual_point_evaluator()
    value_upper_bc = boundary.high.make_virtual_point_evaluator()

    @jit_allocate_out(out_shape=(2, dim_r))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        i = 0
        if r_min == 0:
            # Apply Neumann condition at the origin
            out[0, i] = (arr[1] - arr[0]) * scale_r
        else:
            arr_r_l = value_lower_bc(arr, (i,))
            out[0, i] = (arr[1] - arr_r_l) * scale_r
        out[1, i] = 0  # no angular dependence by definition

        for i in range(1, dim_r - 1):  # iterate inner radial points
            out[0, i] = (arr[i + 1] - arr[i - 1]) * scale_r
            out[1, i] = 0  # no angular dependence by definition

        i = dim_r - 1
        arr_r_h = value_upper_bc(arr, (i,))
        out[0, i] = (arr_r_h - arr[i - 1]) * scale_r
        out[1, i] = 0  # no angular dependence by definition

        return out

    return gradient  # type: ignore


@PolarGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(bcs: Boundaries, central: bool = True) -> Callable:
    """make a discretized gradient squared operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    dr = bcs.grid.discretization[0]

    # prepare boundary values
    boundary = bcs[0]
    value_lower_bc = boundary.low.make_virtual_point_evaluator()
    value_upper_bc = boundary.high.make_virtual_point_evaluator()

    if central:
        # use central differences
        scale = 1 / (2 * dr) ** 2

        @jit_allocate_out(out_shape=(dim_r,))
        def gradient_squared(arr, out=None):
            """ apply squared gradient operator to array `arr` """
            if r_min == 0:
                # Apply Neumann condition at the origin
                out[0] = (arr[1] - arr[0]) ** 2 * scale
            else:
                arr_r_l = value_lower_bc(arr, (0,))
                out[0] = (arr[1] - arr_r_l) ** 2 * scale

            for i in range(1, dim_r - 1):  # iterate inner radial points
                out[i] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

            i = dim_r - 1
            arr_r_h = value_upper_bc(arr, (i,))
            out[i] = (arr_r_h - arr[i - 1]) ** 2 * scale

            return out

    else:
        # use forward and backward differences
        scale = 1 / (2 * dr ** 2)

        @jit_allocate_out(out_shape=(dim_r,))
        def gradient_squared(arr, out=None):
            """ apply squared gradient operator to array `arr` """
            if r_min == 0:
                # Apply Neumann condition at the origin
                out[0] = (arr[1] - arr[0]) ** 2 * scale
            else:
                arr_r_l = value_lower_bc(arr, (0,))
                out[0] = ((arr[1] - arr[0]) ** 2 + (arr[0] - arr_r_l) ** 2) * scale

            for i in range(1, dim_r - 1):  # iterate inner radial points
                term = (arr[i + 1] - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2
                out[i] = term * scale

            i = dim_r - 1
            arr_r_h = value_upper_bc(arr, (i,))
            out[i] = ((arr_r_h - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2) * scale

            return out

    return gradient_squared  # type: ignore


@PolarGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(bcs: Boundaries) -> Callable:
    """make a discretized divergence operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    scale_r = 1 / (2 * dr)

    # prepare boundary values
    boundary = bcs[0]
    value_lower_bc = boundary.low.make_virtual_point_evaluator()
    value_upper_bc = boundary.high.make_virtual_point_evaluator()

    if r_min == 0:

        @jit_allocate_out(out_shape=(dim_r,))
        def divergence(arr, out=None):
            """ apply divergence operator to array `arr` """
            # inner radial boundary condition
            i = 0
            out[i] = (arr[0, 1] + 3 * arr[0, 0]) * scale_r

            for i in range(1, dim_r - 1):  # iterate radial points
                out[i] = (arr[0, i + 1] - arr[0, i - 1]) * scale_r
                out[i] += arr[0, i] / ((i + 0.5) * dr)

            # outer radial boundary condition
            i = dim_r - 1
            arr_r_h = value_upper_bc(arr[0], (i,))
            out[i] = (arr_r_h - arr[0, i - 1]) * scale_r
            out[i] += arr[0, i] / ((i + 0.5) * dr)

            return out

    else:  # r_min > 0

        @jit_allocate_out(out_shape=(dim_r,))
        def divergence(arr, out=None):
            """ apply divergence operator to array `arr` """
            # inner radial boundary condition
            i = 0
            arr_r_l = value_lower_bc(arr[0], (i,))
            out[i] = (arr[0, i + 1] - arr_r_l) * scale_r + arr[0, i] / rs[i]

            for i in range(1, dim_r - 1):  # iterate radial points
                out[i] = (arr[0, i + 1] - arr[0, i - 1]) * scale_r + arr[0, i] / rs[i]

            # outer radial boundary condition
            i = dim_r - 1
            arr_r_h = value_upper_bc(arr[0], (i,))
            out[i] = (arr_r_h - arr[0, i - 1]) * scale_r + arr[0, i] / rs[i]

            return out

    return divergence  # type: ignore


@PolarGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(bcs: Boundaries) -> Callable:
    """make a discretized vector gradient operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(1)

    gradient_r = make_gradient(bcs.extract_component(0))
    gradient_phi = make_gradient(bcs.extract_component(1))

    @jit_allocate_out(out_shape=(2, 2) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        gradient_r(arr[0], out=out[:, 0])
        gradient_phi(arr[1], out=out[:, 1])
        return out

    return vector_gradient  # type: ignore


@PolarGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(bcs: Boundaries) -> Callable:
    """make a discretized tensor divergence operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(1)

    divergence_r = make_divergence(bcs.extract_component(0))
    divergence_phi = make_divergence(bcs.extract_component(1))

    @jit_allocate_out(out_shape=(2,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        divergence_r(arr[0], out=out[0])
        divergence_phi(arr[1], out=out[1])
        return out

    return tensor_divergence  # type: ignore


@fill_in_docstring
def _get_laplace_matrix(bcs):
    """get sparse matrix for laplace operator on a polar grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    scale = 1 / dr ** 2

    matrix = sparse.dok_matrix((dim_r, dim_r))
    vector = sparse.dok_matrix((dim_r, 1))

    for i in range(dim_r):
        matrix[i, i] += -2 * scale
        scale_i = 1 / (2 * rs[i] * dr)

        if i == 0:
            if r_min == 0:
                matrix[i, i + 1] = 2 * scale
                continue  # the special case of the inner boundary is handled
            else:
                const, entries = bcs[0].get_data((-1,))
                factor = scale - scale_i
                vector[i] += const * factor
                for k, v in entries.items():
                    matrix[i, k] += v * factor

        else:
            matrix[i, i - 1] = scale - scale_i

        if i == dim_r - 1:
            const, entries = bcs[0].get_data((dim_r,))
            factor = scale + scale_i
            vector[i] += const * factor
            for k, v in entries.items():
                matrix[i, k] += v * factor

        else:
            matrix[i, i + 1] = scale + scale_i

    return matrix, vector


@PolarGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
@fill_in_docstring
def make_poisson_solver(bcs: Boundaries, method: str = "auto") -> Callable:
    """make a operator that solves Poisson's equation

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)
