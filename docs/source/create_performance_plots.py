#!/usr/bin/env python3
"""
Code for creating performance plots

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os

os.environ["NUMBA_NUM_THREADS"] = "1"  # check single thread performance

import functools
import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


def time_function(func, arg, repeat=3, use_out=False):
    """estimates the computation speed of a function

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


def get_performance_data(periodic=False):
    """obtain the data used in the performance plot

    Args:
        periodic (bool): The boundary conditions of the underlying grid

    Returns:
        dict: The durations of calculating the Laplacian on different grids
        using different methods
    """
    sizes = 2 ** np.arange(3, 13)

    statistics = {}
    for size in display_progress(sizes):
        data = {}
        grid = UnitGrid([size] * 2, periodic=periodic)
        field = ScalarField.random_normal(grid)
        field.set_ghost_cells(bc="auto_periodic_neumann")

        for backend in ["numba", "scipy"]:
            op = grid.make_operator(
                "laplace", bc="auto_periodic_neumann", backend=backend
            )
            data[backend] = time_function(op, field.data)

        op = grid.make_operator_no_bc("laplace", backend="numba")
        data["numba_no_bc"] = time_function(op, field._data_full, use_out=True)

        if opencv_laplace:
            data["opencv"] = time_function(opencv_laplace, field.data)

        statistics[int(size)] = data

    return statistics


def plot_performance(performance_data, title=None):
    """plot the performance data

    Args:
        performance_data: The data obtained from calling
            :func:`get_performance_data`.
        title (str): The title of the plot
    """
    plt.figure(figsize=[4, 3])

    PLOT_DATA = [
        {"key": "scipy", "label": "scipy", "fmt": "C0.-"},
        {"key": "opencv", "label": "opencv", "fmt": "C1.-"},
        {"key": "numba", "label": "py-pde", "fmt": "C2.-"},
        {"key": "numba_no_bc", "label": "py-pde (no BCs)", "fmt": "C2:"},
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
    """run main scripts"""
    data = get_performance_data(periodic=False)
    plot_performance(data, title="2D Laplacian (reflecting BCs)")
    plt.savefig("performance_noflux.pdf", transparent=True)
    plt.savefig("performance_noflux.png", transparent=True, dpi=200)
    plt.close()

    data = get_performance_data(periodic=True)
    plot_performance(data, title="2D Laplacian (periodic BCs)")
    plt.savefig("performance_periodic.pdf", transparent=True)
    plt.savefig("performance_periodic.png", transparent=True, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
