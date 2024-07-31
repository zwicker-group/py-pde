#!/usr/bin/env python3
"""This script creates storage files for backwards compatibility tests."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

import pde


def create_storage_test_resources(path, num):
    """Test storing scalar field as movie."""
    grid = pde.CylindricalSymGrid(3, [1, 2], [2, 2])
    field = pde.ScalarField(grid, [[1, 3], [2, 4]])
    eq = pde.DiffusionPDE()
    info = {"payload": "storage-test"}
    movie_writer = pde.MovieStorage(
        path / f"storage_{num}.avi",
        info=info,
        vmax=4,
        bits_per_channel=16,
        write_times=True,
    )
    file_writer = pde.FileStorage(path / f"storage_{num}.hdf5", info=info)
    interrupts = pde.FixedInterrupts([0.1, 0.7, 2.9])
    eq.solve(
        field,
        t_range=3.5,
        dt=0.1,
        backend="numpy",
        tracker=[movie_writer.tracker(interrupts), file_writer.tracker(interrupts)],
    )


def main():
    """Main function creating all the requirements."""
    root = Path(PACKAGE_PATH)
    create_storage_test_resources(root / "tests" / "storage" / "resources", 2)


if __name__ == "__main__":
    main()
