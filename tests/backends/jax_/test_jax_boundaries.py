"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from pde import ScalarField, UnitGrid
from pde.backends.jax import jax_backend


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    "bc",
    [
        {"type": "derivative", "value": -1},
        {"type": "derivative", "value": 1},
        "extrapolate",
        {"type": "mixed", "value": 4, "const": 1},
        {"type": "mixed", "value": 2, "const": 4},
    ],
)
def test_virtual_points_jax(backend, dim, bc, rng):
    """Test the calculation of virtual points."""
    g = UnitGrid([2] * dim)
    field = ScalarField.random_uniform(g, dtype=np.float32, rng=rng)
    bcs = g.get_boundary_conditions(bc)

    f1 = field.copy()
    f1.set_ghost_cells(bcs)
    f2 = field.copy()
    bc_setter = jax_backend.make_data_setter(g, rank=0, bcs=bc)
    res = backend._apply_native(bc_setter, f2.data, inplace=False)
    if dim == 1:
        np.testing.assert_allclose(f1._data_full, res)
    elif dim == 2:
        np.testing.assert_allclose(f1._data_full[1:-1, :], res[1:-1, :])
        np.testing.assert_allclose(f1._data_full[:, 1:-1], res[:, 1:-1])
    else:
        raise NotImplementedError


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_inhomogeneous_bcs_1d_jax(backend):
    """Test inhomogeneous boundary conditions in 1d grids."""
    g = UnitGrid([2])
    field = ScalarField(g, [1, 1], dtype=np.float32)

    # first order bc
    bc = g.get_boundary_conditions({"value": "x**2"})
    setter = jax_backend.make_data_setter(g, rank=0, bcs=bc)
    data_full = backend._apply_native(setter, field.data)
    assert data_full[-1] == pytest.approx(7.0)
    assert data_full[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_inhomogeneous_bcs_2d_jax(backend):
    """Test inhomogeneous boundary conditions in 2d grids."""
    g = UnitGrid([2, 2])
    f1 = ScalarField(g, "ones", dtype=np.float32)
    f2 = f1.copy()

    # second order bc
    bc = g.get_boundary_conditions({"curvature": "y"})
    f2.set_ghost_cells(bc)
    setter = jax_backend.make_data_setter(g, rank=0, bcs=bc)
    data_full = backend._apply_native(setter, f1.data)
    np.testing.assert_allclose(data_full[1:-1, :], f2._data_full[1:-1, :])
    np.testing.assert_allclose(data_full[:, 1:-1], f2._data_full[:, 1:-1])
