"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, UnitGrid, VectorField
from pde.pdes import (
    helmholtz_decomposition,
    solve_laplace_equation,
    solve_poisson_equation,
)


def test_pde_poisson_solver_1d():
    """Test the poisson solver on 1d grids."""
    # solve Laplace's equation
    grid = UnitGrid([4])
    res = solve_laplace_equation(grid, bc={"x-": {"value": -1}, "x+": {"value": 3}})
    np.testing.assert_allclose(res.data, grid.axes_coords[0] - 1)

    res = solve_laplace_equation(
        grid, bc={"x-": {"value": -1}, "x+": {"derivative": 1}}
    )
    np.testing.assert_allclose(res.data, grid.axes_coords[0] - 1)

    # test Poisson equation with 2nd Order BC
    res = solve_laplace_equation(grid, bc={"x-": {"value": -1}, "x+": "extrapolate"})

    # solve Poisson's equation
    grid = CartesianGrid([[0, 1]], 4)
    field = ScalarField(grid, data=1)

    res = solve_poisson_equation(
        field, bc={"x-": {"value": 1}, "x+": {"derivative": 1}}
    )
    xs = grid.axes_coords[0]
    np.testing.assert_allclose(res.data, 1 + 0.5 * xs**2, rtol=1e-2)

    # test inconsistent problem
    field.data = 1
    with pytest.raises(RuntimeError, match="Neumann"):
        solve_poisson_equation(field, {"derivative": 0})


def test_pde_poisson_solver_2d():
    """Test the poisson solver on 2d grids."""
    grid = CartesianGrid([[0, 2 * np.pi]] * 2, 16)
    bcs = {"x": {"value": "sin(y)"}, "y": {"value": "sin(x)"}}

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
    bcs = {"x": {"value": "sin(y)"}, "y": {"curvature": "sin(x)"}}
    res = solve_laplace_equation(grid, bc=bcs)


def test_helmholtz_decomposition_1d():
    """Test the helmholtz decomposition in 1D."""
    grid = CartesianGrid([[0, 2 * np.pi]], 32, periodic=True)
    field = VectorField.from_expression(grid, ["sin(x)"])
    phi, vec = helmholtz_decomposition(field, bc="auto_periodic_neumann")
    phi_grad = phi.gradient("auto_periodic_neumann")
    np.testing.assert_allclose(field.data, phi_grad.data, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(vec.data, 0, atol=1e-2, rtol=1e-2)


def test_helmholtz_decomposition_2d():
    """Test the helmholtz decomposition in 2D."""
    grid = CartesianGrid([[0, 2 * np.pi]] * 2, 32, periodic=True)
    field = VectorField.from_expression(grid, ["sin(x)", "cos(x)"])
    phi, vec = helmholtz_decomposition(field, bc="auto_periodic_neumann")
    phi_grad = phi.gradient("auto_periodic_neumann")
    assert not np.allclose(field.data, phi_grad.data, atol=1e-2, rtol=1e-2)
    assert not np.allclose(vec.data, 0, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(field.data, (phi_grad + vec).data, atol=1e-2, rtol=1e-2)
