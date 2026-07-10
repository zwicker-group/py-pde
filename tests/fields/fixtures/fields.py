"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import (
    CartesianGrid,
    CylindricalSymGrid,
    FieldCollection,
    PolarSymGrid,
    ScalarField,
    SphericalSymGrid,
    Tensor2Field,
    UnitGrid,
    VectorField,
)


def get_grids():
    """Provide some test grids."""
    grid_list = [PolarSymGrid(3, 4), SphericalSymGrid(3, 4)]
    for periodic in [True, False]:
        grid_list.extend(
            [
                UnitGrid([3], periodic=periodic),
                UnitGrid([3, 3, 3], periodic=periodic),
                CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic),
                CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=periodic),
            ]
        )
    return grid_list


def get_fields():
    """Provide some test fields."""
    grid = CartesianGrid([[0, 2], [-1, 1]], [3, 4], [True, False])
    return [
        ScalarField(UnitGrid([1, 2, 3]), 1),
        VectorField.from_expression(PolarSymGrid(2, 3), ["r**2", "r"]),
        Tensor2Field.random_normal(
            CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=True)
        ),
        FieldCollection([ScalarField(grid, 1), VectorField(grid, 2)]),
    ]


def get_cartesian_grid(dim=2, periodic=True):
    """Return a random Cartesian grid of given dimension."""
    rng = np.random.default_rng(0)
    bounds = [[0, 1 + rng.random()] for _ in range(dim)]
    shape = rng.integers(32, 64, size=dim)
    return CartesianGrid(bounds, shape, periodic=periodic)
