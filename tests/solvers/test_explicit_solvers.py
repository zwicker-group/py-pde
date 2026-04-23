"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from pde import PDE, DiffusionPDE, MemoryStorage, ScalarField, UnitGrid
from pde.pdes import PDEBase
from pde.solvers import (
    Controller,
    EulerSolver,
    ExplicitMPISolver,
    MilsteinSolver,
    RungeKuttaSolver,
)
from pde.tools import mpi

ALL_BACKENDS = ["numpy", "numba", "jax", "torch-cpu", "torch-mps", "torch-cuda"]


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_solvers_simple_fixed(backend):
    """Test explicit solvers."""
    grid = UnitGrid([4])
    xs = grid.axes_coords[0]
    field = ScalarField.from_expression(grid, "x")
    eq = PDE({"c": "c"})

    dt = 0.001

    if mpi.parallel_run:
        solver = ExplicitMPISolver(eq, backend=backend, adaptive=False)
    else:
        solver = EulerSolver(eq, backend=backend, adaptive=False)
    controller = Controller(solver, t_range=10.0, tracker=None)
    res = controller.run(field, dt=dt)

    if mpi.is_main:
        np.testing.assert_allclose(res.data, xs * np.exp(10), rtol=0.1)
        assert solver.info["steps"] == pytest.approx(10 / dt, abs=1)
        assert not solver.info.get("dt_adaptive", False)


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_simple_adaptive(backend):
    """Test explicit solvers."""
    grid = UnitGrid([4])
    y0 = np.array([1e-3, 1e-3, 1e3, 1e3])
    field = ScalarField(grid, y0)
    eq = PDE({"c": "c"})
    dt = 0.1

    args = {"backend": backend, "adaptive": True, "tolerance": 1e-1}
    if mpi.parallel_run:
        solver = ExplicitMPISolver(eq, **args)
    else:
        solver = EulerSolver(eq, **args)
    storage = MemoryStorage()
    controller = Controller(solver, t_range=10.1, tracker=storage.tracker(1.0))
    res = controller.run(field, dt=dt)

    if mpi.is_main:
        np.testing.assert_allclose(res.data, y0 * np.exp(10.1), rtol=0.02)
        assert solver.info["steps"] != pytest.approx(10.1 / dt, abs=1)
        assert solver.info["dt_adaptive"]
        assert solver.info["dt_statistics"]["min"] < 0.0005
        assert np.allclose(storage.times, np.arange(11))


@pytest.mark.parametrize("solver", [EulerSolver, RungeKuttaSolver])
@pytest.mark.parametrize("adaptive", [True, False])
def test_solvers_time_dependent(solver, adaptive):
    """Test explicit solvers with a simple ODE."""
    grid = UnitGrid([1])
    field = ScalarField(grid, 1)
    eq = PDE({"y": "2*sin(t) - y"})

    dt = 1e-3 if solver is EulerSolver else 1e-2

    storage = MemoryStorage()
    solver = solver(eq, adaptive=adaptive)
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


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
@pytest.mark.parametrize("solver", [EulerSolver, MilsteinSolver])
def test_stochastic_solvers(backend, solver, rng):
    """Test simple version of the stochastic solver."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-10)

    solver1 = solver(eq, backend=backend)
    c1 = Controller(solver1, t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    solver2 = solver(seq, backend=backend)
    c2 = Controller(solver2, t_range=1, tracker=None)
    s2 = c2.run(field, dt=1e-3)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-4, atol=1e-4)
    assert not solver1.info["stochastic"]
    assert solver2.info["stochastic"]

    assert not solver1.info["dt_adaptive"]
    assert not solver2.info["dt_adaptive"]


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
@pytest.mark.parametrize("solver", [EulerSolver, MilsteinSolver])
def test_stochastic_solvers_geometric_brownian_motion(backend, solver, rng):
    """Compare stochastic solvers to the analytical GBM moments."""

    class GeometricBrownianMotion(PDE):
        """Simple geometric Brownian motion with multiplicative noise."""

        def __init__(self, mu: float, sigma: float, *, rng):
            # Use a non-zero noise flag so the stochastic solver path is enabled.
            super().__init__({"c": f"{mu} * c"}, noise=1, rng=rng)
            self.mu = float(mu)
            self.sigma = float(sigma)

        def make_noise_variance(self, state, *, backend, ret_diff=False):
            sigma2 = self.sigma**2

            if ret_diff:

                def noise_variance(state_data, t):
                    variance = sigma2 * state_data**2
                    return variance, 2 * sigma2 * state_data

                return noise_variance

            def noise_variance(state_data, t):
                return sigma2 * state_data**2

            return noise_variance

    mu = 0.35
    sigma = 0.25
    initial_value = 1.4
    t_range = 0.5
    dt = 5e-4
    num_realizations = 8192

    eq = GeometricBrownianMotion(mu, sigma, rng=rng)
    field = ScalarField(UnitGrid([num_realizations]), data=initial_value)
    result = eq.solve(
        field,
        t_range=t_range,
        dt=dt,
        solver=solver,
        backend=backend,
        tracker=None,
    )

    data = np.ravel(result.data)
    mean_expected = initial_value * np.exp(mu * t_range)
    var_expected = (
        initial_value**2 * np.exp(2 * mu * t_range) * (np.exp(sigma**2 * t_range) - 1)
    )

    np.testing.assert_allclose(data.mean(), mean_expected, rtol=0.03)
    np.testing.assert_allclose(data.var(), var_expected, rtol=0.1)
    assert eq.diagnostics["solver"]["stochastic"]


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
@pytest.mark.parametrize("solver_cls", [EulerSolver, MilsteinSolver])
@pytest.mark.parametrize("mobility_factor", [0.0, 1.0])
@pytest.mark.parametrize("interpretation", ["ito", "anti-ito"])
def test_stochastic_solver_equilibrium(
    backend, solver_cls, mobility_factor, interpretation, rng
):
    """Test wether conversion to Itô representation is correct."""
    k_val = 0.5  # stiffness of the potential

    class MultiplicativeNoise(PDE):
        """Simple example with potential multiplicative noise."""

        def __init__(self, k: float, *, rng, noise_interpretation):
            # Use a non-zero noise flag so the stochastic solver path is enabled.
            super().__init__(
                {"c": f"-(1 + {mobility_factor} * c**2) * {k} * c"},
                noise=1,
                rng=rng,
                noise_interpretation=noise_interpretation,
            )

        def make_noise_variance(self, state, *, backend, ret_diff=False):
            if ret_diff:

                def noise_variance(state_data, t):
                    mobility = 1 + mobility_factor * state_data**2
                    mobility_diff = 2 * mobility_factor * state_data
                    return 2 * mobility, 2 * mobility_diff  # kBT=1

                return noise_variance

            def noise_variance(state_data, t):
                mobility = 1 + mobility_factor * state_data**2
                return 2 * mobility  # kBT=1

            return noise_variance

    # define a simple test case without spatial diffusivity
    eq = MultiplicativeNoise(k=k_val, rng=rng, noise_interpretation=interpretation)

    # simulate an independent ensemble of points
    field = ScalarField(UnitGrid([1024]))
    solver = solver_cls(eq, backend=backend)
    controller = Controller(solver, t_range=5)
    result = controller.run(field, dt=1e-3)

    # check whether the points exhibit the expected distribution
    sigma = 1 / np.sqrt(k_val)
    test_res = stats.kstest(result.data, "norm", args=(0, sigma))
    if interpretation == "ito" and mobility_factor != 0:
        assert test_res.pvalue < 0.05  # expect different distribution
    else:
        assert test_res.pvalue > 0.05  # expect Gaussian distribution


def test_stochastic_adaptive_solver(caplog, rng):
    """Test using an adaptive, stochastic solver."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE(noise=1e-6)

    solver = EulerSolver(eq, backend="numpy", adaptive=True)
    c = Controller(solver, t_range=1, tracker=None)
    with pytest.raises(RuntimeError):
        c.run(field, dt=1e-2)


@pytest.mark.parametrize("backend", ALL_BACKENDS, indirect=True)
def test_stochastic_solvers_two_interfaces(backend, rng):
    """Compare the two noise interfaces of stochastic solvers."""

    class DiffusionNoisePDE(DiffusionPDE):
        custom_noise = 0.1
        use_noise_variance = False
        use_noise_realization = True

        def make_noise_realization(self, state, backend):
            data_shape = state.data.shape
            scale = np.sqrt(float(self.custom_noise) / state.grid.cell_volumes)

            def noise_realization(state_data, t):
                """Helper function returning a noise realization."""
                return scale * np.random.randn(*data_shape)

            return noise_realization

    eq1 = DiffusionNoisePDE(rng=rng)
    eq2 = DiffusionPDE(noise=0.1, rng=rng)

    field = ScalarField(UnitGrid([16]))
    args = {
        "t_range": 1,
        "dt": 1e-2,
        "solver": "euler",
        "backend": backend,
        "tracker": None,
    }
    if backend.implementation == "torch":
        with pytest.raises(NotImplementedError):
            eq1.solve(field, **args)
        return

    res1 = eq1.solve(field, **args)
    res2 = eq2.solve(field, **args)

    assert res1.fluctuations > 0.1
    assert res2.fluctuations > 0.1

    assert eq1.diagnostics["solver"]["stochastic"]
    assert eq2.diagnostics["solver"]["stochastic"]


@pytest.mark.parametrize(
    "solver", ["adams-bashforth", "crank-nicolson", "runge-kutta", "scipy"]
)
def test_unsupported_stochastic_solvers(solver, rng):
    """Test some solvers that do not support stochasticity."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE(noise=1)

    with pytest.raises(RuntimeError):
        eq.solve(field, 1, solver=solver, tracker=None)


@pytest.mark.parametrize("solver", ["euler", "runge-kutta"])
def test_adaptive_solver_nan(solver):
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
            return state.copy()

    field = ScalarField(UnitGrid([2]))
    eq = MockPDE()
    sol, info = eq.solve(
        field,
        1,
        dt=0.1,
        solver=solver,
        backend="numpy",
        tracker=None,
        adaptive=True,
        ret_info=True,
    )

    np.testing.assert_allclose(sol.data, 0)
    assert info["solver"]["dt_statistics"]["max"] > 0.1
    assert info["solver"]["dt_adaptive"]
