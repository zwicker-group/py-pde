"""
Using an interactive tracker
============================

This example illustrates how a simulation can be analyzed live using the
`napari <https://napari.org>`_ viewer.
"""

import pde


def main():
    grid = pde.UnitGrid([64, 64])
    field = pde.ScalarField.random_uniform(grid, label="Density")

    eq = pde.CahnHilliardPDE()
    eq.solve(field, t_range=1e3, dt=1e-3, tracker=["progress", "interactive"])


if __name__ == "__main__":
    # this safeguard is required since the interactive tracker uses multiprocessing
    main()
