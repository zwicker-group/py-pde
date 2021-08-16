#!/usr/bin/env python3
"""
This script tests the performance of the implementation of different boundary conditions
"""

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE_PATH))

import numpy as np

from pde import ScalarField, UnitGrid
from pde.tools.misc import estimate_computation_speed


def main():
    """main routine testing the performance"""
    print("Reports calls-per-second (larger is better)\n")

    # Cartesian grid with different shapes and boundary conditions
    for size in [32, 512]:
        grid = UnitGrid([size, size], periodic=False)
        print(grid)

        field = ScalarField.random_normal(grid)
        bc_value = np.ones(size)
        result = field.laplace(bc={"value": 1}).data

        for bc in ["scalar", "array", "function", "linked"]:
            if bc == "scalar":
                bcs = {"value": 1}
            elif bc == "array":
                bcs = {"value": bc_value}
            elif bc == "linked":
                bcs = grid.get_boundary_conditions({"value": bc_value})
                for ax, upper in grid._iter_boundaries():
                    bcs[ax][upper].link_value(bc_value)
            elif bc == "function":
                bcs = grid.get_boundary_conditions({"virtual_point": "2 - value"})
            else:
                raise RuntimeError

            # create the operator with these conditions
            laplace = grid.get_operator("laplace", bc=bcs)
            # call once to pre-compile and test result
            np.testing.assert_allclose(laplace(field.data), result)
            # estimate the spped
            speed = estimate_computation_speed(laplace, field.data)
            print(f"{bc:>8s}:{int(speed):>9d}")

        print()


if __name__ == "__main__":
    main()
