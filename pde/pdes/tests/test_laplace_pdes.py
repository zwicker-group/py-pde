"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.pdes import solve_laplace_equation, solve_poisson_equation


def test_pde_poisson_solver_1d():
    """ test the poisson solver on 1d grids """
    # solve Laplace's equation
    grid = UnitGrid([4])
    res = solve_laplace_equation(grid, bc=[{"value": -1}, {"value": 3}])
    np.testing.assert_allclose(res.data, grid.axes_coords[0] - 1)

    res = solve_laplace_equation(grid, bc=[{"value": -1}, {"derivative": 1}])
    np.testing.assert_allclose(res.data, grid.axes_coords[0] - 1)

    # test Poisson equation with 2nd Order BC
    res = solve_laplace_equation(grid, bc=[{"value": -1}, "extrapolate"])

    # solve Poisson's equation
    grid = CartesianGrid([[0, 1]], 4)
    field = ScalarField(grid, data=1)

    res = solve_poisson_equation(field, bc=[{"value": 1}, {"derivative": 1}])
    xs = grid.axes_coords[0]
    np.testing.assert_allclose(res.data, 1 + 0.5 * xs ** 2, rtol=1e-2)

    # test inconsistent problem
    field.data = 1
    with pytest.raises(RuntimeError, match="Neumann"):
        solve_poisson_equation(field, {"derivative": 0})


def test_pde_poisson_solver_2d():
    """ test the poisson solver on 2d grids """
    grid = CartesianGrid([[0, 2 * np.pi]] * 2, 16)
    bcs = [{"value": "sin(y)"}, {"value": "sin(x)"}]

    # solve Laplace's equation
    res = solve_laplace_equation(grid, bcs)
    xs = grid.cell_coords[..., 0]
    ys = grid.cell_coords[..., 1]

    # analytical solution was obtained with Mathematica
    expect = (
        np.cosh(np.pi - ys) * np.sin(xs) + np.cosh(np.pi - xs) * np.sin(ys)
    ) / np.cosh(np.pi)
    np.testing.assert_allclose(res.data, expect, atol=1e-2, rtol=1e-2)

    # test more complex case for exceptions
    bcs = [{"value": "sin(y)"}, {"curvature": "sin(x)"}]
    res = solve_laplace_equation(grid, bc=bcs)
