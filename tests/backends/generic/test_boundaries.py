"""Generic tests for boundary conditions, applied to multiple backends.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
import pytest

from pde import ScalarField, UnitGrid

ALL_BACKENDS = ["numba", "jax", "torch-cpu", "torch-mps", "torch-cuda"]


def _apply_ghost_cells(backend, field: ScalarField, bcs):
    """Apply ghost cells using the backend's native implementation.

    Returns the full data array (including ghost cells) after applying the BCs.
    """
    f_copy = field.copy()
    if backend.name == "jax":
        from pde.backends.jax import jax_backend as _jax_backend

        setter = _jax_backend.make_data_setter(field.grid, rank=field.rank, bcs=bcs)
        backend._apply_operator(setter, field.data, out=f_copy._data_full)

    elif backend.name.startswith("torch"):
        from pde.backends.torch._boundaries import GhostCellSetter as _GhostCellSetter

        setter = _GhostCellSetter(bcs=bcs, dtype=field.dtype).to(backend.device)
        backend._apply_operator(setter, f_copy._data_full, out=f_copy._data_full)

    else:
        # numba, numpy, scipy - use make_ghost_cell_setter
        setter = backend.make_ghost_cell_setter(bcs)
        setter(f_copy._data_full)

    return f_copy._data_full


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
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
def test_virtual_points(backend, dim, bc, rng):
    """Test the calculation of virtual points."""
    g = UnitGrid([2] * dim)
    field = ScalarField.random_uniform(g, dtype=np.float32, rng=rng)
    bcs = g.get_boundary_conditions(bc)

    # Reference using Python implementation
    f_ref = field.copy()
    f_ref.set_ghost_cells(bcs)

    # Apply ghost cells using the backend
    res = _apply_ghost_cells(backend, field, bcs)

    if dim == 1:
        np.testing.assert_allclose(f_ref._data_full, res)
    elif dim == 2:
        np.testing.assert_allclose(f_ref._data_full[1:-1, :], res[1:-1, :])
        np.testing.assert_allclose(f_ref._data_full[:, 1:-1], res[:, 1:-1])
    else:
        raise NotImplementedError


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_inhomogeneous_bcs_1d(backend):
    """Test inhomogeneous boundary conditions in 1d grids."""
    g = UnitGrid([2])
    field = ScalarField(g, [1, 1], dtype=np.float32)
    bcs = g.get_boundary_conditions({"value": "x**2"})

    data_full = _apply_ghost_cells(backend, field, bcs)
    assert data_full[-1] == pytest.approx(7.0)
    assert data_full[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_inhomogeneous_bcs_2d(backend):
    """Test inhomogeneous boundary conditions in 2d grids."""
    g = UnitGrid([2, 2])
    f1 = ScalarField(g, "ones", dtype=np.float32)
    f2 = f1.copy()
    bcs = g.get_boundary_conditions({"curvature": "y"})

    # Reference
    f2.set_ghost_cells(bcs)

    # Backend
    res = _apply_ghost_cells(backend, f1, bcs)
    np.testing.assert_allclose(res[1:-1, :], f2._data_full[1:-1, :])
    np.testing.assert_allclose(res[:, 1:-1], f2._data_full[:, 1:-1])


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    "bc",
    [
        {"type": "virtual_point", "value": "x**2"},
        {"type": "value_expression", "value": "x**2"},
        {"type": "derivative_expression", "value": "x"},
        {"type": "virtual_point", "value": "value + x"},
    ],
)
def test_expression_bc(backend, dim, bc, rng):
    """Test ExpressionBC (virtual point depends on coordinates or field value)."""
    from pde.grids.boundaries.local import ExpressionBC

    g = UnitGrid([2] * dim)
    field = ScalarField.random_uniform(g, dtype=np.float32, rng=rng)
    bcs = g.get_boundary_conditions(bc)

    # verify these are indeed ExpressionBC instances
    for bc_axis in bcs:
        for bc_local in bc_axis:
            assert isinstance(bc_local, ExpressionBC)

    # reference using Python implementation
    f_ref = field.copy()
    f_ref.set_ghost_cells(bcs)

    # apply ghost cells using the backend
    res = _apply_ghost_cells(backend, field, bcs)

    if dim == 1:
        np.testing.assert_allclose(
            np.array(f_ref._data_full, dtype=np.float32),
            np.array(res, dtype=np.float32),
            rtol=1e-5,
        )
    elif dim == 2:
        np.testing.assert_allclose(
            np.array(f_ref._data_full[1:-1, :], dtype=np.float32),
            np.array(res[1:-1, :], dtype=np.float32),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(f_ref._data_full[:, 1:-1], dtype=np.float32),
            np.array(res[:, 1:-1], dtype=np.float32),
            rtol=1e-5,
        )
    else:
        raise NotImplementedError
