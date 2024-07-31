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


def iter_grids():
    """Generator providing some test grids."""
    for periodic in [True, False]:
        yield UnitGrid([3], periodic=periodic)
        yield UnitGrid([3, 3, 3], periodic=periodic)
        yield CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic)
        yield CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=periodic)
    yield PolarSymGrid(3, 4)
    yield SphericalSymGrid(3, 4)


def iter_fields():
    """Generator providing some test fields."""
    yield ScalarField(UnitGrid([1, 2, 3]), 1)
    yield VectorField.from_expression(PolarSymGrid(2, 3), ["r**2", "r"])
    yield Tensor2Field.random_normal(
        CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=True)
    )

    grid = CartesianGrid([[0, 2], [-1, 1]], [3, 4], [True, False])
    yield FieldCollection([ScalarField(grid, 1), VectorField(grid, 2)])


def get_cartesian_grid(dim=2, periodic=True):
    """Return a random Cartesian grid of given dimension."""
    rng = np.random.default_rng(0)
    bounds = [[0, 1 + rng.random()] for _ in range(dim)]
    shape = rng.integers(32, 64, size=dim)
    return CartesianGrid(bounds, shape, periodic=periodic)
