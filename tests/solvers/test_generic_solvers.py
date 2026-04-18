"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging

import numpy as np
import pytest

from pde import (
    PDE,
    CartesianGrid,
    DiffusionPDE,
    FieldCollection,
    MemoryStorage,
    ScalarField,
    UnitGrid,
)
from pde.solvers import (
    AdamsBashforthSolver,
    Controller,
    CrankNicolsonSolver,
    EulerSolver,
    ImplicitSolver,
    RungeKuttaSolver,
    ScipySolver,
    registered_solvers,
)

SOLVER_CLASSES = [
    AdamsBashforthSolver,
    CrankNicolsonSolver,
    EulerSolver,
    ImplicitSolver,
    RungeKuttaSolver,
    ScipySolver,
]

NOT_SUPPORTED = {
    "jax": {CrankNicolsonSolver, ImplicitSolver},
    "torch": {
        AdamsBashforthSolver,
        CrankNicolsonSolver,
        ImplicitSolver,
        RungeKuttaSolver,
    },
}

ALL_BACKENDS = ["numpy", "numba", "jax", "torch-cpu", "torch-mps", "torch-cuda"]


def test_solver_registration():
    """Test solver registration."""
    solvers = registered_solvers()
    assert "euler" in solvers
    assert "implicit" in solvers
    assert "crank-nicolson" in solvers
    assert "scipy" in solvers


def test_solver_in_pde_class(rng):
    """Test whether solver instances can be used in pde instances."""
    field = ScalarField.random_uniform(UnitGrid([16, 16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    eq.solve(field, t_range=1, solver=ScipySolver, tracker=None)


@pytest.mark.parametrize("solver_class", SOLVER_CLASSES)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_complex(solver_class, backend):
    """Test solvers with a complex PDE."""
    r = FieldCollection.scalar_random_uniform(2, UnitGrid([3]), labels=["a", "b"])
    c = r["a"] + 1j * r["b"]
    assert c.is_complex

    # assume c = a + i * b
    eq_c = PDE({"c": "-I * laplace(c)"})
    eq_r = PDE({"a": "laplace(b)", "b": "-laplace(a)"})
    res_r = eq_r.solve(r, t_range=1e-2, dt=1e-3, backend="numpy", tracker=None)
    exp_c = res_r[0].data + 1j * res_r[1].data

    solver = solver_class(eq_c, backend=backend)
    controller = Controller(solver, t_range=1e-2, tracker=None)
    res_c = controller.run(c, dt=1e-3)
    np.testing.assert_allclose(res_c.data, exp_c, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("solver_class", [EulerSolver, RungeKuttaSolver])
def test_basic_adaptive_solver(solver_class):
    """Test basic adaptive solvers."""
    grid = UnitGrid([4])
    y0 = np.array([1e-3, 1e-3, 1e3, 1e3])
    field = ScalarField(grid, y0)
    eq = PDE({"c": "c"})

    dt = 0.1

    solver = solver_class(eq, adaptive=True, tolerance=1e-1)
    storage = MemoryStorage()
    controller = Controller(solver, t_range=10.1, tracker=storage.tracker(1.0))
    res = controller.run(field, dt=dt)

    np.testing.assert_allclose(res.data, y0 * np.exp(10.1), rtol=0.02)
    assert solver.info["steps"] != pytest.approx(10.1 / dt, abs=1)
    assert solver.info["dt_adaptive"]
    if solver_class is EulerSolver:
        assert solver.info["dt_statistics"]["min"] < 0.0005
    elif solver_class is RungeKuttaSolver:
        assert solver.info["dt_statistics"]["min"] < 0.05
    assert np.allclose(storage.times, np.arange(11))


@pytest.mark.parametrize("solver", SOLVER_CLASSES)
@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_deterministic_solver_backend_support(solver, backend, caplog):
    """Test all backends for all deterministic solvers with fixed dt."""
    eq = DiffusionPDE(noise=0)
    grid = CartesianGrid([[-6, 6]], 32)
    field = ScalarField.from_expression(grid, "heaviside(x)")

    args = {"t_range": 1, "solver": solver, "backend": backend, "tracker": None}

    if solver in NOT_SUPPORTED.get(backend.implementation, set()):
        # solver is not supported by backend
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises((NotImplementedError, TypeError)) as exc_info,
        ):
            result = eq.solve(field, **args)
        if exc_info.type is not NotImplementedError:
            # if a random error is raised,
            assert "not supported by backend" in caplog.text

    else:
        # solver is supported by backend
        result = eq.solve(field, **args)
        expect = ScalarField.from_expression(grid, "0.5 + 0.5 * erf(x/2)")
        np.testing.assert_allclose(result.data, expect.data, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("solver", [EulerSolver, RungeKuttaSolver])
@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_adaptive_solver_backend_support(solver, backend, caplog):
    """Test all backends for all deterministic solvers with adaptive dt."""
    eq = DiffusionPDE(noise=0)
    grid = CartesianGrid([[-6, 6]], 32)
    field = ScalarField.from_expression(grid, "heaviside(x)")

    args = {
        "t_range": 1,
        "adaptive": True,
        "solver": solver,
        "backend": backend,
        "tracker": None,
    }

    if solver in NOT_SUPPORTED.get(backend.implementation, set()):
        # solver is not supported by backend
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises((NotImplementedError, TypeError)) as exc_info,
        ):
            result = eq.solve(field, **args)
        if exc_info.type is not NotImplementedError:
            # if a random error is raised,
            assert "not supported by backend" in caplog.text

    else:
        # solver is supported by backend
        result = eq.solve(field, **args)
        expect = ScalarField.from_expression(grid, "0.5 + 0.5 * erf(x/2)")
        np.testing.assert_allclose(result.data, expect.data, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("solver", [EulerSolver, ImplicitSolver])
@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_stochastic_solver_backend_support(solver, backend, caplog, rng):
    """Test all backends for all deterministic solvers with fixed dt."""
    eq = DiffusionPDE(noise=1e-4, rng=rng)
    grid = CartesianGrid([[-6, 6]], 32)
    field = ScalarField.from_expression(grid, "heaviside(x)")

    args = {
        "t_range": 1,
        "dt": 1e-3,
        "solver": solver,
        "backend": backend,
        "tracker": None,
    }

    if solver in NOT_SUPPORTED.get(backend.implementation, set()):
        # solver is not supported by backend
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises((NotImplementedError, TypeError)) as exc_info,
        ):
            result = eq.solve(field, **args)
        if exc_info.type is not NotImplementedError:
            # if a random error is raised,
            assert "not supported by backend" in caplog.text

    else:
        # solver is supported by backend
        result = eq.solve(field, **args)
        expect = ScalarField.from_expression(grid, "0.5 + 0.5 * erf(x/2)")
        np.testing.assert_allclose(result.data, expect.data, atol=0.1, rtol=1e-2)
