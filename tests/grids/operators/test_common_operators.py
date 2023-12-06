"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField
from pde.grids.operators import cartesian as ops


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_make_derivative(ndim, axis, rng):
    """test the _make_derivative function"""
    periodic = random.choice([True, False])
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add, rng=rng)

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    grad = field.gradient(bcs)
    for method in ["central", "forward", "backward"]:
        msg = f"method={method}, periodic={periodic}"
        diff = ops._make_derivative(grid, axis=axis, method=method)
        res = field.copy()
        res.data[:] = 0
        field.set_ghost_cells(bcs)
        diff(field._data_full, out=res.data)
        np.testing.assert_allclose(
            grad.data[axis], res.data, atol=0.1, rtol=0.1, err_msg=msg
        )


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_make_derivative2(ndim, axis, rng):
    """test the _make_derivative2 function"""
    periodic = random.choice([True, False])
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add, rng=rng)

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    grad = field.gradient(bcs)[axis]
    grad2 = grad.gradient(bcs)[axis]

    diff = ops._make_derivative2(grid, axis=axis)
    res = field.copy()
    res.data[:] = 0
    field.set_ghost_cells(bcs)
    diff(field._data_full, out=res.data)
    np.testing.assert_allclose(grad2.data, res.data, atol=0.1, rtol=0.1)
