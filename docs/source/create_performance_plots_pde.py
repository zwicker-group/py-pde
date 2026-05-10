#!/usr/bin/env python3
"""Code for creating plots showing performance of PDEs.

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
import json
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

import pde
from pde import CahnHilliardPDE, ScalarField, TrackerBase, UnitGrid

# determine path of the cache for the ground truth of simulations
RESULT_CACHE = Path(__file__).resolve().parent / "_cache" / "performance_pde"


class RuntimeTacker(TrackerBase):
    def __init__(self, interrupts):
        super().__init__(interrupts=interrupts)
        self.start_time = time.monotonic()
        self.data: list[tuple[float, float]] = []

    def initialize(self, field, info):
        return super().initialize(field, info)

    def handle(self, field, t) -> None:
        """Handle data supplied to this tracker.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        self.data.append((t, time.monotonic() - self.start_time))


def calculate_single_run(
    backend: str, size: int, *, output: str | None = None
) -> list[tuple[float, float]]:
    """Calculate performance data for a particular configuration.

    Args:
        backend (str):
            Backend to test
        size (int):
            Dimension of the grid
        output (str):
            Filename to which the final frame should be written
    """
    print(f"Run simulation for {backend=} and {size=}")
    rng = np.random.default_rng(0)
    grid = UnitGrid([size] * 2, periodic=True)
    field = ScalarField.random_uniform(grid, rng=rng)
    eq = CahnHilliardPDE()
    runtime = RuntimeTacker(pde.ConstantInterrupts(1e2, 1))
    res = eq.solve(field, t_range=1e3 + 1, dt=1e-2, backend=backend, tracker=runtime)
    if output:
        res.plot(filename=output, action="close")
    return runtime.data


def get_single_run(backend: str, size: int) -> list[tuple[float, float]] | None:
    """Get performance data for a particular configuration.

    Args:
        backend (str):
            Backend to test
        size (int):
            Dimension of the grid
    """
    cache_file = RESULT_CACHE / f"{backend}_{size}.json"
    img_file = RESULT_CACHE / f"{backend}_{size}.pdf"

    if cache_file.exists():
        # read performance data
        with cache_file.open() as f:
            return json.load(f)["runtime"]

    # calculate  performance data
    try:
        runtime = calculate_single_run(backend, size, output=img_file)
    except RuntimeError:
        return None

    data = {
        "runtime": runtime,
        "version": pde.__version__,
        "date": datetime.now(timezone.utc).isoformat(),
    }

    # write  performance data
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump(data, f)

    return runtime


def collect_performance_data(size: int) -> dict[list[tuple[float, float]]]:
    """Obtain the data used in the performance plot.

    Args:
        size (int):
            Dimension of the grid

    Returns:
        dict: The durations of calculating the Laplacian on different grids
        using different methods
    """
    return {
        backend: get_single_run(backend, size)
        for backend in ["numba", "torch:cpu", "torch:cuda", "jax:cpu", "jax:cuda"]
    }


def plot_performance(performance_data):
    """Plot the performance data.

    Args:
        performance_data:
            The data obtained from calling :func:`get_performance_data`.
    """
    plt.figure(figsize=[4, 3])

    PLOT_DATA = [
        # {"key": "numpy", "label": "numpy", "fmt": "C0-"},
        {"key": "numba", "label": "numba", "fmt": "C1-"},
        {"key": "torch:cpu", "label": "torch:cpu", "fmt": "C2-"},
        {"key": "torch:cuda", "label": "torch:cuda", "fmt": "C2:"},
        {"key": "jax:cpu", "label": "jax:cpu", "fmt": "C3-"},
        {"key": "jax:cuda", "label": "jax:cuda", "fmt": "C3:"},
    ]

    xmax = 0
    for plot in PLOT_DATA:
        data = performance_data[plot["key"]]
        if data is None:
            continue
        data = np.asarray(data)
        xmax = max(xmax, data[-1, 0])
        plt.plot(data[:, 0], data[:, 1], plot["fmt"], label=plot["label"])

    plt.xlim(0, xmax)
    plt.xlabel("Simulation time")
    plt.ylabel("Runtime [s]")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()


def main():
    """Run main scripts."""
    data = collect_performance_data(size=256)
    plot_performance(data)
    plt.savefig("performance_cahn_hilliard.pdf", transparent=True)
    plt.savefig("performance_cahn_hilliard.png", transparent=True, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
