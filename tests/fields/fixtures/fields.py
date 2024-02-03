"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import (
    CartesianGrid,
    CylindricalSymGrid,
    PolarSymGrid,
    SphericalSymGrid,
    UnitGrid,
)


def iter_grids():
    """generator providing some test grids"""
    for periodic in [True, False]:
        yield UnitGrid([3], periodic=periodic)
        yield UnitGrid([3, 3, 3], periodic=periodic)
        yield CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic)
        yield CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=periodic)
    yield PolarSymGrid(3, 4)
    yield SphericalSymGrid(3, 4)


def get_cartesian_grid(dim=2, periodic=True):
    """return a random Cartesian grid of given dimension"""
    rng = np.random.default_rng(0)
    bounds = [[0, 1 + rng.random()] for _ in range(dim)]
    shape = rng.integers(32, 64, size=dim)
    return CartesianGrid(bounds, shape, periodic=periodic)
