"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


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
