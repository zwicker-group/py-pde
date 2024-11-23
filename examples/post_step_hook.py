"""
Post-step hook function
=======================

Demonstrate the simple hook function in :class:`~pde.pdes.PDE`, which is called after
each time step and may modify the state and abort the simulation.
"""

from pde import PDE, ScalarField, UnitGrid


def post_step_hook(state_data, t):
    """Helper function called after every time step."""
    state_data[24:40, 24:40] = 1  # set central region to given value

    if t > 1e3:
        raise StopIteration  # abort simulation at given time


eq = PDE({"c": "laplace(c)"}, post_step_hook=post_step_hook)
state = ScalarField(UnitGrid([64, 64]))
result = eq.solve(state, dt=0.1, t_range=1e4)
result.plot()
