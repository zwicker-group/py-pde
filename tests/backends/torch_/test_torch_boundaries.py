"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import numpy as np
import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)


from pde import ScalarField, UnitGrid
from pde.backends.torch._boundaries import GhostCellSetter


@pytest.mark.parametrize(
    "backend", ["torch-cpu", "torch-mps", "torch-cuda"], indirect=True
)
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
def test_virtual_points_torch(backend, dim, bc, rng):
    """Test the calculation of virtual points."""
    g = UnitGrid([2] * dim)
    field = ScalarField.random_uniform(g, dtype=np.float32, rng=rng)
    bcs = g.get_boundary_conditions(bc)

    f1 = field.copy()
    f1.set_ghost_cells(bcs)
    f2 = field.copy()
    bc_setter = GhostCellSetter(bcs=bcs, dtype=f1.dtype).to(backend.device)
    backend._apply_native(bc_setter, f2._data_full, inplace=True)
    if dim == 1:
        np.testing.assert_allclose(f1._data_full, f2._data_full)
    elif dim == 2:
        np.testing.assert_allclose(f1._data_full[1:-1, :], f2._data_full[1:-1, :])
        np.testing.assert_allclose(f1._data_full[:, 1:-1], f2._data_full[:, 1:-1])
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "backend", ["torch-cpu", "torch-mps", "torch-cuda"], indirect=True
)
def test_inhomogeneous_bcs_1d_torch(backend):
    """Test inhomogeneous boundary conditions in 1d grids."""
    g = UnitGrid([2])
    field = ScalarField(g, [1, 1], dtype=np.float32)

    # first order bc
    bc = g.get_boundary_conditions({"value": "x**2"})
    setter = GhostCellSetter(bcs=bc, dtype=field.dtype).to(backend.device)
    backend._apply_native(setter, field._data_full, inplace=True)
    assert field._data_full[-1] == pytest.approx(7.0)
    assert field._data_full[0] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "backend", ["torch-cpu", "torch-mps", "torch-cuda"], indirect=True
)
def test_inhomogeneous_bcs_2d_torch(backend):
    """Test inhomogeneous boundary conditions in 2d grids."""
    g = UnitGrid([2, 2])
    f1 = ScalarField(g, "ones", dtype=np.float32)
    f2 = f1.copy()

    # second order bc
    bc = g.get_boundary_conditions({"curvature": "y"})
    f2.set_ghost_cells(bc)
    setter = GhostCellSetter(bcs=bc, dtype=f2.dtype).to(backend.device)
    backend._apply_native(setter, f1._data_full, inplace=True)
    np.testing.assert_allclose(f1._data_full[1:-1, :], f2._data_full[1:-1, :])
    np.testing.assert_allclose(f1._data_full[:, 1:-1], f2._data_full[:, 1:-1])
