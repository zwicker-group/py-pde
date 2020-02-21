#!/usr/bin/env python3

from pde.common import *


eq = KPZInterfacePDE(noise=1)              # define the SDE
grid = UnitGrid([32])                      # generate grid
state = ScalarField.random_harmonic(grid)  # generate initial condition

storage = MemoryStorage()
eq.solve(state, t_range=10, dt=0.01, tracker=storage.tracker(1))
plot_kymograph(storage, show=True)
