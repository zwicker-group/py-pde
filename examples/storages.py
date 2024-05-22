"""
Storage examples
================

This example shows how to use :mod:`~pde.storage` to store data persistently.
"""

from pde import AllenCahnPDE, FileStorage, MovieStorage, ScalarField, UnitGrid

# initialize the model
state = ScalarField.random_uniform(UnitGrid([128, 128]), -0.01, 0.01)
eq = AllenCahnPDE()

# initialize empty storages
file_write = FileStorage("allen_cahn.hdf")
movie_write = MovieStorage("allen_cahn.avi", vmin=-1, vmax=1)

# store trajectory in storage
final_state = eq.solve(
    state,
    t_range=100,
    adaptive=True,
    tracker=[file_write.tracker(2), movie_write.tracker(1)],
)

# read storage and plot last frame
movie_read = MovieStorage("allen_cahn.avi")
movie_read[-1].plot()
