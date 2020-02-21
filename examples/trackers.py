#!/usr/bin/env python3

from pde.common import *

eq = DiffusionPDE()                                 # define the pde
grid = UnitGrid([16, 16])                           # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

storage = MemoryStorage()

trackers = [
    'progress',                    # show progress bar during simulation
    'steady_state',                # abort when steady state is reached
    storage.tracker(interval=1),   # store data every simulation time unit
    PlotTracker(show=True),        # show images during simulation
    # print some output every 5 real seconds:
    PrintTracker(interval=RealtimeIntervals(duration=5))
    
]

eq.solve(state, 10, dt=0.1, tracker=trackers)

storage[0].plot(show=True)
