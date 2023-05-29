"""
Use multiprocessing via MPI
===========================

Use multiple cores to solve a PDE. The implementation here uses the `Message Passing 
Interface (MPI) <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_, and the
script thus needs to be run using :code:`mpiexec -n 2 python mpi_parallel_run.py`, where
`2` denotes the number of cores used. Note that macOS might require an additional hint
on how to connect the processes. The following line might work:
    `mpiexec -n 2 -host localhost:2 python3 mpi_parallel_run.py`

Such parallel simulations need extra care, since multiple instances of the same program
are started. In particular, in the example below, the initial state is created on all
cores. However, only the state of the first core will actually be used and distributed
automatically by `py-pde`. Note that also only the first (or main) core will run the
trackers and receive the result of the simulation. On all other cores, the simulation
result will be `None`.
"""

from pde import DiffusionPDE, ScalarField, UnitGrid

grid = UnitGrid([64, 64])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

eq = DiffusionPDE(diffusivity=0.1)  # define the pde
result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

if result is not None:  # check whether we are on the main core
    result.plot()
