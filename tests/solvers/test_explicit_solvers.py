"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, DiffusionPDE, MemoryStorage, ScalarField, UnitGrid
from pde.pdes import PDEBase
from pde.solvers import Controller, ExplicitMPISolver, ExplicitSolver
from pde.tools import mpi


@pytest.mark.multiprocessing
@pytest.mark.parametrize("scheme", ["euler", "runge-kutta"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_simple_fixed(scheme, backend):
    """Test explicit solvers."""
    grid = UnitGrid([4])
    xs = grid.axes_coords[0]
    field = ScalarField.from_expression(grid, "x")
    eq = PDE({"c": "c"})

    dt = 0.001 if scheme == "euler" else 0.01

    if mpi.parallel_run:
        solver = ExplicitMPISolver(eq, scheme=scheme, backend=backend, adaptive=False)
    else:
        solver = ExplicitSolver(eq, scheme=scheme, backend=backend, adaptive=False)
    controller = Controller(solver, t_range=10.0, tracker=None)
    res = controller.run(field, dt=dt)

    if mpi.is_main:
        np.testing.assert_allclose(res.data, xs * np.exp(10), rtol=0.1)
        assert solver.info["steps"] == pytest.approx(10 / dt, abs=1)
        assert not solver.info.get("dt_adaptive", False)


@pytest.mark.multiprocessing
@pytest.mark.parametrize("scheme", ["euler", "runge-kutta"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_simple_adaptive(scheme, backend):
    """Test explicit solvers."""
    grid = UnitGrid([4])
    y0 = np.array([1e-3, 1e-3, 1e3, 1e3])
    field = ScalarField(grid, y0)
    eq = PDE({"c": "c"})

    dt = 0.1 if scheme == "euler" else 1

    args = {"scheme": scheme, "backend": backend, "adaptive": True, "tolerance": 1e-1}
    if mpi.parallel_run:
        solver = ExplicitMPISolver(eq, **args)
    else:
        solver = ExplicitSolver(eq, **args)
    storage = MemoryStorage()
    controller = Controller(solver, t_range=10.1, tracker=storage.tracker(1.0))
    res = controller.run(field, dt=dt)

    if mpi.is_main:
        np.testing.assert_allclose(res.data, y0 * np.exp(10.1), rtol=0.02)
        assert solver.info["steps"] != pytest.approx(10.1 / dt, abs=1)
        assert solver.info["dt_adaptive"]
        if scheme == "euler":
            assert solver.info["dt_statistics"]["min"] < 0.0005
        else:
            assert solver.info["dt_statistics"]["min"] < 0.03
        assert np.allclose(storage.times, np.arange(11))


@pytest.mark.parametrize("scheme", ["euler", "runge-kutta"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_solvers_time_dependent(scheme, adaptive):
    """Test explicit solvers with a simple ODE."""
    grid = UnitGrid([1])
    field = ScalarField(grid, 1)
    eq = PDE({"y": "2*sin(t) - y"})

    dt = 1e-3 if scheme == "euler" else 1e-2

    storage = MemoryStorage()
    solver = ExplicitSolver(eq, scheme=scheme, adaptive=adaptive)
    controller = Controller(solver, t_range=20.0, tracker=storage.tracker(1))
    controller.run(field, dt=dt)

    ts = np.ravel(storage.times)
    expect = 2 * np.exp(-ts) - np.cos(ts) + np.sin(ts)
    np.testing.assert_allclose(np.ravel(storage.data), expect, atol=0.05)

    if adaptive:
        assert solver.info["steps"] < 20 / dt
    else:
        assert solver.info["steps"] == pytest.approx(20 / dt, abs=1)
    assert solver.info.get("dt_adaptive", False) == adaptive


@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_stochastic_solvers(backend, rng):
    """Test simple version of the stochastic solver."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-10)

    solver1 = ExplicitSolver(eq, backend=backend)
    c1 = Controller(solver1, t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    solver2 = ExplicitSolver(seq, backend=backend)
    c2 = Controller(solver2, t_range=1, tracker=None)
    s2 = c2.run(field, dt=1e-3)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-4, atol=1e-4)
    assert not solver1.info["stochastic"]
    assert solver2.info["stochastic"]

    assert not solver1.info.get("dt_adaptive", False)
    assert not solver2.info.get("dt_adaptive", False)


def test_stochastic_adaptive_solver(caplog, rng):
    """Test using an adaptive, stochastic solver."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE(noise=1e-6)

    solver = ExplicitSolver(eq, backend="numpy", adaptive=True)
    c = Controller(solver, t_range=1, tracker=None)
    with pytest.raises(RuntimeError):
        c.run(field, dt=1e-2)


def test_unsupported_stochastic_solvers(rng):
    """Test some solvers that do not support stochasticity."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE(noise=1)

    with pytest.raises(RuntimeError):
        eq.solve(field, 1, solver="explicit", scheme="runge-kutta", tracker=None)
    with pytest.raises(RuntimeError):
        eq.solve(field, 1, solver="scipy", scheme="runge-kutta", tracker=None)


@pytest.mark.parametrize("scheme", ["euler", "runge-kutta"])
def test_adaptive_solver_nan(scheme):
    """Test whether the adaptive solver can treat nans."""
    # Note for programmer: A similar test for the `numba` backend is difficult to
    # implement, since we only want to fail very rarely. We tried doing it with random
    # failure, but this often resulted in hitting the minimal time step.

    class MockPDE(PDEBase):
        """Simple PDE which returns NaN every 5 evaluations."""

        evaluations = 0

        def evolution_rate(self, state, t):
            MockPDE.evaluations += 1
            if MockPDE.evaluations == 2:
                return ScalarField(state.grid, data=np.nan)
            else:
                return state.copy()

    field = ScalarField(UnitGrid([2]))
    eq = MockPDE()
    sol, info = eq.solve(
        field,
        1,
        dt=0.1,
        solver="explicit",
        scheme=scheme,
        tracker=None,
        adaptive=True,
        ret_info=True,
    )

    np.testing.assert_allclose(sol.data, 0)
    assert info["solver"]["dt_statistics"]["max"] > 0.1
    assert info["solver"]["dt_adaptive"]
