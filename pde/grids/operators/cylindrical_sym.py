r"""
This module implements differential operators on cylindrical grids 

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

from ... import config
from ...tools.docstrings import fill_in_docstring
from ...tools.numba import jit_allocate_out, nb
from ...tools.typing import OperatorType
from ..boundaries import Boundaries
from ..cylindrical import CylindricalSymGrid


@CylindricalSymGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(bcs: Boundaries) -> OperatorType:
    """make a discretized laplace operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(0)
    boundary_r, boundary_z = bcs

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    dr_2, dz_2 = 1 / bcs.grid.discretization ** 2

    value_r_outer = boundary_r.high.make_virtual_point_evaluator()
    region_z = boundary_z.make_region_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(dim_r, dim_z))
    def laplace(arr, out=None):
        """apply laplace operator to array `arr`"""
        for j in nb.prange(0, dim_z):  # iterate axial points
            # inner radial boundary condition
            i = 0
            arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
            out[i, j] = (
                2 * (arr[i + 1, j] - arr_c) * dr_2
                + (arr_z_l - 2 * arr_c + arr_z_h) * dz_2
            )

            if dim_r == 1:
                continue  # deal with singular radial dimension

            for i in range(1, dim_r - 1):  # iterate radial points
                arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
                arr_r_l, arr_r_h = arr[i - 1, j], arr[i + 1, j]
                out[i, j] = (
                    (arr_r_h - 2 * arr_c + arr_r_l) * dr_2
                    + (arr_r_h - arr_r_l) / (2 * i + 1) * dr_2
                    + (arr_z_l - 2 * arr_c + arr_z_h) * dz_2
                )

            # outer radial boundary condition
            i = dim_r - 1
            arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
            arr_r_l, arr_r_h = arr[i - 1, j], value_r_outer(arr, (i, j))
            out[i, j] = (
                (arr_r_h - 2 * arr_c + arr_r_l) * dr_2
                + (arr_r_h - arr_r_l) / (2 * i + 1) * dr_2
                + (arr_z_l - 2 * arr_c + arr_z_h) * dz_2
            )
        return out

    return laplace  # type: ignore


@CylindricalSymGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(bcs: Boundaries) -> OperatorType:
    """make a discretized gradient operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(0)
    boundary_r, boundary_z = bcs

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    scale_r, scale_z = 1 / (2 * bcs.grid.discretization)

    value_r_outer = boundary_r.high.make_virtual_point_evaluator()
    region_z = boundary_z.make_region_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(3, dim_r, dim_z))
    def gradient(arr, out=None):
        """apply gradient operator to array `arr`"""
        for j in nb.prange(0, dim_z):  # iterate axial points
            # inner radial boundary condition
            i = 0
            arr_z_l, _, arr_z_h = region_z(arr, (i, j))
            out[0, i, j] = (arr[1, i] - arr[0, i]) * scale_r
            out[1, i, j] = (arr_z_h - arr_z_l) * scale_z
            out[2, i, j] = 0  # no phi dependence by definition

            for i in range(1, dim_r - 1):  # iterate radial points
                arr_z_l, _, arr_z_h = region_z(arr, (i, j))
                out[0, i, j] = (arr[i + 1, j] - arr[i - 1, j]) * scale_r
                out[1, i, j] = (arr_z_h - arr_z_l) * scale_z
                out[2, i, j] = 0  # no phi dependence by definition

            # outer radial boundary condition
            i = dim_r - 1
            arr_z_l, _, arr_z_h = region_z(arr, (i, j))
            arr_r_h = value_r_outer(arr, (i, j))
            out[0, i, j] = (arr_r_h - arr[i - 1, j]) * scale_r
            out[1, i, j] = (arr_z_h - arr_z_l) * scale_z
            out[2, i, j] = 0  # no phi dependence by definition

        return out

    return gradient  # type: ignore


@CylindricalSymGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(bcs: Boundaries, central: bool = True) -> OperatorType:
    """make a discretized gradient squared operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

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
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(0)
    boundary_r, boundary_z = bcs

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape

    value_r_outer = boundary_r.high.make_virtual_point_evaluator()
    region_z = boundary_z.make_region_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    if central:
        # use central differences
        scale_r, scale_z = 1 / (2 * bcs.grid.discretization) ** 2

        @jit_allocate_out(parallel=parallel, out_shape=(dim_r, dim_z))
        def gradient_squared(arr, out=None):
            """apply gradient operator to array `arr`"""
            for j in nb.prange(0, dim_z):  # iterate axial points
                # inner radial boundary condition (Neumann condition)
                i = 0
                arr_z_l, _, arr_z_h = region_z(arr, (i, j))
                term_r = (arr[1, j] - arr[0, j]) ** 2
                term_z = (arr_z_h - arr_z_l) ** 2
                out[i, j] = term_r * scale_r + term_z * scale_z

                for i in range(1, dim_r - 1):  # iterate radial points
                    arr_z_l, _, arr_z_h = region_z(arr, (i, j))
                    term_r = (arr[i + 1, j] - arr[i - 1, j]) ** 2
                    term_z = (arr_z_h - arr_z_l) ** 2
                    out[i, j] = term_r * scale_r + term_z * scale_z

                # outer radial boundary condition
                i = dim_r - 1
                arr_z_l, _, arr_z_h = region_z(arr, (i, j))
                arr_r_h = value_r_outer(arr, (i, j))
                term_r = (arr_r_h - arr[i - 1, j]) ** 2
                term_z = (arr_z_h - arr_z_l) ** 2
                out[i, j] = term_r * scale_r + term_z * scale_z

            return out

    else:
        # use forward and backward differences
        scale_r, scale_z = 1 / (2 * bcs.grid.discretization ** 2)

        @jit_allocate_out(parallel=parallel, out_shape=(dim_r, dim_z))
        def gradient_squared(arr, out=None):
            """apply gradient operator to array `arr`"""
            for j in nb.prange(0, dim_z):  # iterate axial points
                # inner radial boundary condition (Neumann condition)
                i = 0
                arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
                term_r = (arr[1, j] - arr[0, j]) ** 2
                term_z = (arr_z_h - arr_c) ** 2 + (arr_c - arr_z_l) ** 2
                out[i, j] = term_r * scale_r + term_z * scale_z

                for i in range(1, dim_r - 1):  # iterate radial points
                    arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
                    term_r = (arr[i + 1, j] - arr_c) ** 2 + (arr_c - arr[i - 1, j]) ** 2
                    term_z = (arr_z_h - arr_c) ** 2 + (arr_c - arr_z_l) ** 2
                    out[i, j] = term_r * scale_r + term_z * scale_z

                # outer radial boundary condition
                i = dim_r - 1
                arr_z_l, arr_c, arr_z_h = region_z(arr, (i, j))
                arr_r_h = value_r_outer(arr, (i, j))
                term_r = (arr_r_h - arr_c) ** 2 + (arr_c - arr[i - 1, j]) ** 2
                term_z = (arr_z_h - arr_c) ** 2 + (arr_c - arr_z_l) ** 2
                out[i, j] = term_r * scale_r + term_z * scale_z

            return out

    return gradient_squared  # type: ignore


@CylindricalSymGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(bcs: Boundaries) -> OperatorType:
    """make a discretized divergence operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(0)
    boundary_r, boundary_z = bcs

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    rs = bcs.grid.axes_coords[0]
    der_r = boundary_r.make_derivative_evaluator()
    der_z = boundary_z.make_derivative_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(dim_r, dim_z))
    def divergence(arr, out=None):
        """apply divergence operator to array `arr`"""
        arr_r, arr_z = arr[0], arr[1]

        for j in nb.prange(dim_z):  # iterate axial points
            for i in range(dim_r):  # iterate radial points
                out[i, j] = (
                    arr_r[i, j] / rs[i] + der_z(arr_z, (i, j)) + der_r(arr_r, (i, j))
                )

        return out

    return divergence  # type: ignore


@CylindricalSymGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(bcs: Boundaries) -> OperatorType:
    """make a discretized vector gradient operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(1)

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    rs = bcs.grid.axes_coords[0]

    # handle the boundary conditions
    boundary_r, boundary_z = bcs
    deriv_r_r = boundary_r.extract_component(0).make_derivative_evaluator()
    deriv_r_z = boundary_r.extract_component(1).make_derivative_evaluator()
    deriv_r_φ = boundary_r.extract_component(2).make_derivative_evaluator()
    deriv_z_r = boundary_z.extract_component(0).make_derivative_evaluator()
    deriv_z_z = boundary_z.extract_component(1).make_derivative_evaluator()
    deriv_z_φ = boundary_z.extract_component(2).make_derivative_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(3, 3) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """apply gradient operator to array `arr`"""
        # assign aliases
        arr_r, arr_z, arr_φ = arr
        out_rr, out_rz, out_rφ = out[0, 0], out[0, 1], out[0, 2]
        out_zr, out_zz, out_zφ = out[1, 0], out[1, 1], out[1, 2]
        out_φr, out_φz, out_φφ = out[2, 0], out[2, 1], out[2, 2]

        for j in nb.prange(dim_z):  # iterate axial points
            for i in range(dim_r):  # iterate radial points
                out_rr[i, j] = deriv_r_r(arr_r, (i, j))
                out_φr[i, j] = deriv_r_φ(arr_φ, (i, j))
                out_zr[i, j] = deriv_r_z(arr_z, (i, j))

                out_rφ[i, j] = -arr_φ[i, j] / rs[i]
                out_φφ[i, j] = arr_r[i, j] / rs[i]
                out_zφ[i, j] = 0

                out_rz[i, j] = deriv_z_r(arr_r, (i, j))
                out_φz[i, j] = deriv_z_φ(arr_φ, (i, j))
                out_zz[i, j] = deriv_z_z(arr_z, (i, j))

        return out

    return vector_gradient  # type: ignore


@CylindricalSymGrid.register_operator("vector_laplace", rank_in=1, rank_out=1)
@fill_in_docstring
def make_vector_laplace(bcs: Boundaries) -> OperatorType:
    """make a discretized vector laplace operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(1)

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    rs = bcs.grid.axes_coords[0]
    dr = bcs.grid.discretization[0]
    s1, s2 = 1 / (2 * dr), 1 / dr ** 2

    # handle the boundary conditions
    boundary_r, boundary_z = bcs
    region_r_r = boundary_r.extract_component(0).make_region_evaluator()
    region_r_z = boundary_r.extract_component(1).make_region_evaluator()
    region_r_φ = boundary_r.extract_component(2).make_region_evaluator()
    deriv_zz_r = boundary_z.extract_component(0).make_derivative_evaluator(order=2)
    deriv_zz_z = boundary_z.extract_component(1).make_derivative_evaluator(order=2)
    deriv_zz_φ = boundary_z.extract_component(2).make_derivative_evaluator(order=2)

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(3,) + bcs.grid.shape)
    def vector_laplace(arr, out=None):
        """apply vector laplace operator to array `arr`"""
        # assign aliases
        arr_r, arr_z, arr_φ = arr
        out_r, out_z, out_φ = out

        for j in nb.prange(dim_z):  # iterate axial points
            for i in range(dim_r):  # iterate radial points
                f_r_l, f_r_m, f_r_h = region_r_r(arr_r, (i, j))
                out_r[i, j] = (
                    deriv_zz_r(arr_r, (i, j))
                    - f_r_m / rs[i] ** 2
                    + (f_r_h - f_r_l) * s1 / rs[i]
                    + (f_r_h - 2 * f_r_m + f_r_l) * s2
                )

                f_φ_l, f_φ_m, f_φ_h = region_r_φ(arr_φ, (i, j))
                out_φ[i, j] = (
                    deriv_zz_φ(arr_φ, (i, j))
                    - f_φ_m / rs[i] ** 2
                    + (f_φ_h - f_φ_l) * s1 / rs[i]
                    + (f_φ_h - 2 * f_φ_m + f_φ_l) * s2
                )

                f_z_l, f_z_m, f_z_h = region_r_z(arr_z, (i, j))
                out_z[i, j] = (
                    deriv_zz_z(arr_z, (i, j))
                    + (f_z_h - f_z_l) * s1 / rs[i]
                    + (f_z_h - 2 * f_z_m + f_z_l) * s2
                )

        return out

    return vector_laplace  # type: ignore


@CylindricalSymGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(bcs: Boundaries) -> OperatorType:
    """make a discretized tensor divergence operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, CylindricalSymGrid)
    bcs.check_value_rank(1)

    # calculate preliminary quantities
    dim_r, dim_z = bcs.grid.shape
    rs = bcs.grid.axes_coords[0]

    # handle the boundary conditions
    boundary_r, boundary_z = bcs
    deriv_r_r = boundary_r.extract_component(0).make_derivative_evaluator()
    deriv_r_z = boundary_r.extract_component(1).make_derivative_evaluator()
    deriv_r_φ = boundary_r.extract_component(2).make_derivative_evaluator()
    deriv_z_r = boundary_z.extract_component(0).make_derivative_evaluator()
    deriv_z_z = boundary_z.extract_component(1).make_derivative_evaluator()
    deriv_z_φ = boundary_z.extract_component(2).make_derivative_evaluator()

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.parallel_threshold"]

    @jit_allocate_out(parallel=parallel, out_shape=(3,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """apply tensor divergence operator to array `arr`"""
        # assign aliases
        arr_rr, arr_rz, arr_rφ = arr[0, 0], arr[0, 1], arr[0, 2]
        arr_zr, arr_zz, _ = arr[1, 0], arr[1, 1], arr[1, 2]
        arr_φr, arr_φz, arr_φφ = arr[2, 0], arr[2, 1], arr[2, 2]
        out_r, out_z, out_φ = out

        for j in nb.prange(dim_z):  # iterate axial points
            for i in range(dim_r):  # iterate radial points
                derivs = deriv_z_r(arr_rz, (i, j)) + deriv_r_r(arr_rr, (i, j))
                out_r[i, j] = derivs + (arr_rr[i, j] - arr_φφ[i, j]) / rs[i]

                derivs = deriv_z_φ(arr_φz, (i, j)) + deriv_r_φ(arr_φr, (i, j))
                out_φ[i, j] = derivs + (arr_rφ[i, j] + arr_φr[i, j]) / rs[i]

                derivs = deriv_z_z(arr_zz, (i, j)) + deriv_r_z(arr_zr, (i, j))
                out_z[i, j] = derivs + arr_zr[i, j] / rs[i]

        return out

    return tensor_divergence  # type: ignore
