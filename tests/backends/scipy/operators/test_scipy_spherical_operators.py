"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import ScalarField, SphericalSymGrid, solve_poisson_equation
from pde.backends.scipy.operators.common import make_laplace_from_matrix
from pde.backends.scipy.operators.spherical_sym import _get_laplace_matrix


@pytest.mark.parametrize("grid", [SphericalSymGrid(4, 8), SphericalSymGrid([2, 4], 8)])
@pytest.mark.parametrize("bc_val", ["auto_periodic_neumann", {"value": 1}])
def test_poisson_solver_spherical(grid, bc_val, rng):
    """Test the poisson solver on Spherical grids."""
    bcs = grid.get_boundary_conditions(bc_val)
    d = ScalarField.random_uniform(grid, rng=rng)
    d -= d.average  # balance the right hand side
    sol = solve_poisson_equation(d, bcs)
    test = sol.laplace(bcs)
    np.testing.assert_allclose(
        test.data, d.data, err_msg=f"bcs={bc_val}, grid={grid}", rtol=1e-6
    )


@pytest.mark.parametrize("r_inner", [0, 1])
def test_laplace_matrix(r_inner, rng):
    """Test laplace operator implemented using matrix multiplication."""
    grid = SphericalSymGrid((r_inner, 2), 16)
    if r_inner == 0:
        bcs = grid.get_boundary_conditions("neumann")
    else:
        bcs = grid.get_boundary_conditions({"value": "sin(r)"})
    laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = field.laplace(bcs)
    res2 = laplace(field.data)

    np.testing.assert_allclose(res1.data, res2)
