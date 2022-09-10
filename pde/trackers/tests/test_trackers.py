"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import pickle

import numpy as np
import pytest

from pde import Controller, ExplicitSolver, MemoryStorage, ScalarField, UnitGrid
from pde.pdes import AllenCahnPDE, CahnHilliardPDE, DiffusionPDE
from pde.tools.misc import module_available
from pde.trackers import get_named_trackers, trackers
from pde.trackers.base import TrackerBase
from pde.trackers.interrupts import ConstantInterrupts
from pde.visualization.movies import Movie


def test_plot_tracker(tmp_path):
    """test whether the plot tracker creates files without errors"""
    output_file = tmp_path / "img.png"

    def get_title(state, t):
        return f"{state.integral:g} at {t:g}"

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid)
    eq = DiffusionPDE()
    tracker = trackers.PlotTracker(
        output_file=output_file, title=get_title, interval=0.1, show=False
    )

    eq.solve(state, t_range=0.5, dt=0.005, tracker=tracker, backend="numpy")

    assert output_file.stat().st_size > 0


@pytest.mark.skipif(not Movie.is_available(), reason="no ffmpeg")
def test_plot_movie_tracker(tmp_path):
    """test whether the plot tracker creates files without errors"""
    output_file = tmp_path / "movie.mov"

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid)
    eq = DiffusionPDE()
    tracker = trackers.PlotTracker(movie=output_file, interval=0.1, show=False)

    eq.solve(state, t_range=0.5, dt=0.005, tracker=tracker, backend="numpy")

    assert output_file.stat().st_size > 0


def test_simple_progress():
    """simple test for basic progress bar"""
    pbar = trackers.ProgressTracker(interval=1)
    field = ScalarField(UnitGrid([3]))
    pbar.initialize(field)
    pbar.handle(field, 2)
    pbar.finalize()


def test_trackers():
    """test whether simple trackers can be used"""
    times = []

    def store_time(state, t):
        times.append(t)

    def get_data(state):
        return {"integral": state.integral}

    devnull = open(os.devnull, "w")
    data = trackers.DataTracker(get_data, interval=0.1)
    tracker_list = [
        trackers.PrintTracker(interval=0.1, stream=devnull),
        trackers.CallbackTracker(store_time, interval=0.1),
        None,  # should be ignored
        data,
    ]
    if module_available("matplotlib"):
        tracker_list.append(trackers.PlotTracker(interval=0.1, show=False))

    grid = UnitGrid([16, 16])
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    eq = DiffusionPDE()
    eq.solve(state, t_range=1, dt=0.005, tracker=tracker_list)

    devnull.close()

    assert times == data.times
    if module_available("pandas"):
        df = data.dataframe
        np.testing.assert_allclose(df["time"], times)
        np.testing.assert_allclose(df["integral"], state.integral)


def test_callback_tracker():
    """test trackers that support a callback"""
    data = []

    def store_mean_data(state):
        data.append(state.average)

    def get_mean_data(state):
        return state.average

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    eq = DiffusionPDE()
    data_tracker = trackers.DataTracker(get_mean_data, interval=0.1)
    callback_tracker = trackers.CallbackTracker(store_mean_data, interval=0.1)
    tracker_list = [data_tracker, callback_tracker]
    eq.solve(state, t_range=0.5, dt=0.005, tracker=tracker_list, backend="numpy")

    np.testing.assert_array_equal(data, data_tracker.data)

    data = []

    def store_time(state, t):
        data.append(t)

    def get_time(state, t):
        return t

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    eq = DiffusionPDE()
    data_tracker = trackers.DataTracker(get_time, interval=0.1)
    tracker_list = [trackers.CallbackTracker(store_time, interval=0.1), data_tracker]
    eq.solve(state, t_range=0.5, dt=0.005, tracker=tracker_list, backend="numpy")

    ts = np.arange(0, 0.55, 0.1)
    np.testing.assert_allclose(data, ts, atol=1e-2)
    np.testing.assert_allclose(data_tracker.data, ts, atol=1e-2)


def test_data_tracker(tmp_path):
    """test the DataTracker"""
    field = ScalarField(UnitGrid([4, 4]))
    eq = DiffusionPDE()

    path = tmp_path / "test_data_tracker.pickle"
    data1 = trackers.DataTracker(lambda f: f.average, filename=path)
    data2 = trackers.DataTracker(lambda f: {"avg": f.average, "int": f.integral})
    eq.solve(field, 10, tracker=[data1, data2])

    with path.open("br") as fp:
        time, data = pickle.load(fp)
    np.testing.assert_allclose(time, np.arange(11))
    assert isinstance(data, list)
    assert len(data) == 11

    assert path.stat().st_size > 0


def test_steady_state_tracker():
    """test the SteadyStateTracker"""
    storage = MemoryStorage()
    c0 = ScalarField.from_expression(UnitGrid([5]), "sin(x)")
    eq = DiffusionPDE()
    tracker = trackers.SteadyStateTracker(atol=0.05, rtol=0.05, progress=True)
    eq.solve(c0, 1e4, dt=0.1, tracker=[tracker, storage.tracker(interval=1e2)])
    assert len(storage) < 20  # finished early


def test_small_tracker_dt():
    """test the case where the dt of the tracker is smaller than the dt
    of the simulation"""
    storage = MemoryStorage()
    eq = DiffusionPDE()
    c0 = ScalarField.random_uniform(UnitGrid([4, 4]), 0.1, 0.2)
    eq.solve(
        c0, 1e-2, dt=1e-3, method="explicit", tracker=storage.tracker(interval=1e-4)
    )
    assert len(storage) == 11


def test_runtime_tracker():
    """test the RuntimeTracker"""
    s = ScalarField.random_uniform(UnitGrid([128]))
    tracker = trackers.RuntimeTracker("0:01")
    sol = ExplicitSolver(DiffusionPDE())
    con = Controller(sol, t_range=1e4, tracker=["progress", tracker])
    con.run(s, dt=1e-3)


def test_consistency_tracker():
    """test the ConsistencyTracker"""
    s = ScalarField.random_uniform(UnitGrid([128]))
    sol = ExplicitSolver(DiffusionPDE(1e3))
    con = Controller(sol, t_range=1e5, tracker=["consistency"])
    with np.errstate(all="ignore"):
        con.run(s, dt=1)
    assert con.info["t_final"] < con.info["t_end"]


def test_material_conservation_tracker():
    """test the MaterialConservationTracker"""
    state = ScalarField.random_uniform(UnitGrid([8, 8]), 0, 1)

    solver = ExplicitSolver(CahnHilliardPDE())
    controller = Controller(solver, t_range=1, tracker=["material_conservation"])
    controller.run(state, dt=1e-3)
    assert controller.info["t_final"] >= 1

    solver = ExplicitSolver(AllenCahnPDE())
    controller = Controller(solver, t_range=1, tracker=["material_conservation"])
    controller.run(state, dt=1e-3)
    assert controller.info["t_final"] <= 1


def test_get_named_trackers():
    """test the get_named_trackers function"""
    for name, cls in get_named_trackers().items():
        assert isinstance(name, str)
        tracker = TrackerBase.from_data(name)
        assert isinstance(tracker, cls)


def test_double_tracker():
    """simple test for using a custom tracker twice"""
    interval = ConstantInterrupts(1)
    times1, times2 = [], []
    t1 = trackers.CallbackTracker(lambda s, t: times1.append(t), interval=interval)
    t2 = trackers.CallbackTracker(lambda s, t: times2.append(t), interval=interval)

    field = ScalarField.random_uniform(UnitGrid([3]))
    DiffusionPDE().solve(field, t_range=4, dt=0.1, tracker=[t1, t2])

    np.testing.assert_allclose(times1, np.arange(5))
    np.testing.assert_allclose(times2, np.arange(5))
