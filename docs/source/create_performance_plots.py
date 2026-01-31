#!/usr/bin/env python3
"""Code for creating performance plots.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys
from pathlib import Path

# determine path of the `py-pde` package
PACKAGE_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PATH))

# disable multithreading
import os

from pde import config

os.environ["NUMBA_NUM_THREADS"] = "1"
config["backend.numba.multithreading"] = "never"

# import remaining packages
import functools
import json
import timeit
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pde
from pde import ScalarField, UnitGrid
from pde.tools.misc import estimate_computation_speed
from pde.tools.output import display_progress

try:
    import cv2
except ImportError:
    print("Warning: OpenCV is not available and will thus not appear in the comparison")
    opencv_laplace = None
else:
    opencv_laplace = functools.partial(
        cv2.Laplacian, ddepth=cv2.CV_64F, borderType=cv2.BORDER_REFLECT
    )
# determine path of the cache for the ground truth of simulations
GROUND_TRUTH_CACHE = Path(__file__).resolve().parent / "_cache" / "performance_data"


def time_function(func, arg, repeat: int = 3, use_out: bool = False) -> float:
    """Estimates the computation speed of a function.

    Args:
        func (callable): The function to test
        arg: The single argument on which the function will be estimate
        repeat (int): How often the function is tested
        use_out (bool): Whether to supply the `out` argument to the function

    Returns:
        float: Estimated duration of calling the function a single time
    """
    if use_out:
        out = np.empty(tuple(s - 2 for s in arg.shape))
        number = int(estimate_computation_speed(func, arg, out))
        func = functools.partial(func, arg, out)
    else:
        number = int(estimate_computation_speed(func, arg))
        func = functools.partial(func, arg)
    return min(timeit.repeat(func, number=number, repeat=repeat)) / number


def calculate_single_run(backend: str, size: int, periodic=False) -> float:
    """Calculate performance data for a particular configuration.

    Args:
        backend (str):
            Backend to test
        size (int):
            Dimension of the grid
        periodic (bool):
            Whether the grid is periodic or not
    """
    rng = np.random.default_rng(0)
    grid = UnitGrid([size] * 2, periodic=periodic)
    field = ScalarField.random_normal(grid, rng=rng)
    field.set_ghost_cells(bc="auto_periodic_neumann")

    if backend in {"numba", "torch", "scipy"}:
        op = grid.make_operator("laplace", bc="auto_periodic_neumann", backend=backend)
        return time_function(op, field.data)

    if backend == "numba_no_bc":
        op = grid.make_operator_no_bc("laplace", backend="numba")
        return time_function(op, field._data_full, use_out=True)

    if backend == "opencv" or backend == "cv":
        return time_function(opencv_laplace, field.data)

    msg = f"Unknown backend `{backend}`"
    raise ValueError(msg)


def get_single_run(backend: str, size: int, periodic=False) -> float:
    """Get performance data for a particular configuration.

    Args:
        backend (str):
            Backend to test
        size (int):
            Dimension of the grid
        periodic (bool):
            Whether the grid is periodic or not
    """
    cache_file = GROUND_TRUTH_CACHE / f"{backend}_{size}_{periodic}.json"

    if cache_file.exists():
        # read performance data
        with cache_file.open() as f:
            runtime = json.load(f)["runtime"]

    else:
        # calculate  performance data
        runtime = calculate_single_run(backend, size, periodic)
        data = {
            "runtime": runtime,
            "version": pde.__version__,
            "date": datetime.now(timezone.utc).isoformat(),
        }

        # write  performance data
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w") as f:
            return json.dump(data, f)

    return runtime


def collect_performance_data(periodic=False) -> dict[int, dict[str, float]]:
    """Obtain the data used in the performance plot.

    Args:
        periodic (bool): The boundary conditions of the underlying grid

    Returns:
        dict: The durations of calculating the Laplacian on different grids
        using different methods
    """
    sizes = 2 ** np.arange(3, 13)

    statistics = {}
    for size in display_progress(sizes):
        statistics[int(size)] = {
            backend: get_single_run(backend, size, periodic)
            for backend in ["numba", "numba_no_bc", "torch", "scipy", "opencv"]
        }

    return statistics


def plot_performance(performance_data, title=None):
    """Plot the performance data.

    Args:
        performance_data: The data obtained from calling
            :func:`get_performance_data`.
        title (str): The title of the plot
    """
    plt.figure(figsize=[4, 3])

    PLOT_DATA = [
        {"key": "numba", "label": "py-pde [numba]", "fmt": "C0.-"},
        {"key": "numba_no_bc", "label": "", "fmt": "C0:"},
        {"key": "torch", "label": "py-pde [torch:cpu]", "fmt": "C1.-"},
        {"key": "opencv", "label": "opencv", "fmt": "C2.-"},
        {"key": "scipy", "label": "scipy", "fmt": "C3.-"},
    ]

    sizes = np.array(sorted(performance_data.keys()))
    grid_sizes = sizes**2

    for plot in PLOT_DATA:
        data = np.array([performance_data[size][plot["key"]] for size in sizes])
        plt.loglog(grid_sizes, data, plot["fmt"], label=plot["label"])

    plt.xlim(grid_sizes[0], grid_sizes[-1])
    plt.xlabel("Number of grid points")
    plt.ylabel("Runtime [ms]")
    plt.legend(loc="best")

    # fix ticks of y-axis
    locmaj = mpl.ticker.LogLocator(base=10, numticks=12)
    plt.gca().xaxis.set_major_locator(locmaj)

    if title:
        plt.title(title)

    plt.tight_layout()


def main():
    """Run main scripts."""
    data = collect_performance_data(periodic=False)
    plot_performance(data, title="2D Laplacian (reflecting BCs)")
    plt.savefig("performance_noflux.pdf", transparent=True)
    plt.savefig("performance_noflux.png", transparent=True, dpi=200)
    plt.close()

    data = collect_performance_data(periodic=True)
    plot_performance(data, title="2D Laplacian (periodic BCs)")
    plt.savefig("performance_periodic.pdf", transparent=True)
    plt.savefig("performance_periodic.png", transparent=True, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
