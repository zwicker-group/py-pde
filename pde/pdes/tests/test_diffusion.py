"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from pde import CartesianGrid, DiffusionPDE, ScalarField, UnitGrid


def test_diffusion_single():
    """ test some methods of the simple diffusion model """
    eq = DiffusionPDE()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)
    assert not eq.explicit_time_dependence

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid)

    field = eq.evolution_rate(state)
    assert isinstance(field, ScalarField)
    assert field.grid == grid


def test_simple_diffusion_value():
    """ test a simple diffusion equation with constant boundaries """
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1)
    b_l = {"type": "value", "value": 0}
    b_r = {"type": "value", "value": 1}
    pde = DiffusionPDE(bc=[b_l, b_r])
    sol, info = pde.solve(c, t_range=1, dt=0.001, tracker=None, ret_info=True)
    assert isinstance(info, dict)
    np.testing.assert_allclose(sol.data, grid.axes_coords[0], rtol=5e-3)


def test_simple_diffusion_flux_right():
    """ test a simple diffusion equation with flux boundary on the right """
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1)
    b_l = {"type": "value", "value": 0}
    b_r = {"type": "derivative", "value": 3}
    pde = DiffusionPDE(bc=[b_l, b_r])
    sol = pde.solve(c, t_range=5, dt=0.001, tracker=None)
    np.testing.assert_allclose(sol.data, 3 * grid.axes_coords[0], rtol=5e-3)


def test_simple_diffusion_flux_left():
    """ test a simple diffusion equation with flux boundary on the left """
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1)
    b_l = {"type": "derivative", "value": 2}
    b_r = {"type": "value", "value": 0}
    pde = DiffusionPDE(bc=[b_l, b_r])
    sol = pde.solve(c, t_range=5, dt=0.001, tracker=None)
    np.testing.assert_allclose(sol.data, 2 - 2 * grid.axes_coords[0], rtol=5e-3)
