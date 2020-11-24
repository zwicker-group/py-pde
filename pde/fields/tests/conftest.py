"""
Created on Nov 24, 2020

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from pde import CartesianGrid, CylindricalGrid, PolarGrid, SphericalGrid, UnitGrid


def iter_grids():
    """ generator providing some test grids """
    for periodic in [True, False]:
        yield UnitGrid([3], periodic=periodic)
        yield UnitGrid([3, 3, 3], periodic=periodic)
        yield CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic)
        yield CylindricalGrid(3, [-1, 2], [7, 8], periodic_z=periodic)
    yield PolarGrid(3, 4)
    yield SphericalGrid(3, 4)


def pytest_generate_tests(metafunc):
    """py test hook that creates tests with all grids from iter_grids when the special
    fixtures `example_grid` or `example_grid_nd` are used."""
    if "example_grid" in metafunc.fixturenames:
        metafunc.parametrize("example_grid", iter_grids())

    if "example_grid_nd" in metafunc.fixturenames:
        grids = (grid for grid in iter_grids() if grid.num_axes > 1)
        metafunc.parametrize("example_grid_nd", grids)
