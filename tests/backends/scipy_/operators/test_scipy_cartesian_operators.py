"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, UnitGrid, solve_poisson_equation
from pde.backends.scipy.operators.cartesian import _get_laplace_matrix
from pde.backends.scipy.operators.common import make_laplace_from_matrix


@pytest.mark.parametrize(
    "grid", [UnitGrid([12]), CartesianGrid([(0, 1), (4, 5.5)], 8), UnitGrid([3, 3, 3])]
)
@pytest.mark.parametrize("bc_val", ["auto_periodic_neumann", {"value": "sin(x)"}])
def test_poisson_solver_cartesian(grid, bc_val, rng):
    """Test the poisson solver on cartesian grids."""
    bcs = grid.get_boundary_conditions(bc_val)
    d = ScalarField.random_uniform(grid, rng=rng)
    d -= d.average  # balance the right hand side
    sol = solve_poisson_equation(d, bcs)
    test = sol.laplace(bcs)
    np.testing.assert_allclose(
        test.data, d.data, err_msg=f"bcs={bc_val}, grid={grid}", rtol=1e-6
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_laplace_matrix(ndim, rng):
    """Test laplace operator implemented using matrix multiplication."""
    periodic = [False]
    bc = {"x": {"value": "sin(x)"}}
    if ndim >= 2:
        periodic.append(True)
        bc["y"] = "periodic"
    if ndim >= 3:
        periodic.append(False)
        bc["z"] = "derivative"
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    bcs = grid.get_boundary_conditions(bc)
    laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = field.laplace(bcs)
    res2 = laplace(field.data)

    np.testing.assert_allclose(res1.data, res2)
