"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CylindricalSymGrid, ScalarField, solve_poisson_equation
from pde.backends.scipy.operators.common import make_laplace_from_matrix
from pde.backends.scipy.operators.cylindrical_sym import _get_laplace_matrix


@pytest.mark.parametrize("r_inner", [0, 1])
def test_poisson_solver_cylindrical(r_inner, rng):
    """Test the poisson solver on Cylindrical grids."""
    grid = CylindricalSymGrid((r_inner, 2), (2.5, 4.3), 16)
    if r_inner == 0:
        bcs = {"r": "neumann", "z": {"value": "cos(r) + z"}}
    else:
        bcs = {"r": {"value": "sin(r)"}, "z": {"derivative": "cos(r) + z"}}
    d = ScalarField.random_uniform(grid, rng=rng)
    d -= d.average  # balance the right hand side
    sol = solve_poisson_equation(d, bcs)
    test = sol.laplace(bcs)
    np.testing.assert_allclose(
        test.data, d.data, err_msg=f"bcs={bcs}, grid={grid}", rtol=1e-6
    )


@pytest.mark.parametrize("r_inner", [0, 1])
def test_laplace_matrix(r_inner, rng):
    """Test laplace operator implemented using matrix multiplication."""
    grid = CylindricalSymGrid((r_inner, 2), (2.5, 4.3), 16)
    if r_inner == 0:
        bcs = {"r": "neumann"}
    else:
        bcs = {"r": {"value": "sin(r)"}}
    bcs["z"] = {"derivative": "cos(r) + z"}
    bcs = grid.get_boundary_conditions(bcs)
    laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = field.laplace(bcs)
    res2 = laplace(field.data)

    np.testing.assert_allclose(res1.data, res2)
