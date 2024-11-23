"""
Post-step hook function in a custom class
=========================================

The hook function created by :meth:`~pde.pdes.PDEBase.make_post_step_hook` is called
after each time step. The function can modify the state, keep track of additional
information, and abort the simulation.
"""

from pde import PDEBase, ScalarField, UnitGrid


class CustomPDE(PDEBase):
    def make_post_step_hook(self, state):
        """Create a hook function that is called after every time step."""

        def post_step_hook(state_data, t, post_step_data):
            """Limit state 1 and abort when standard deviation exceeds 1."""
            i = state_data > 1  # get violating entries
            overshoot = (state_data[i] - 1).sum()  # get total correction
            state_data[i] = 1  # limit data entries
            post_step_data += overshoot  # accumulate total correction
            if post_step_data > 400:
                # Abort simulation when correction exceeds 400
                # Note that the `post_step_data` of the previous step will be returned.
                raise StopIteration

        return post_step_hook, 0.0  # hook function and initial value for data

    def evolution_rate(self, state, t=0):
        """Evaluate the right hand side of the evolution equation."""
        return state.__class__(state.grid, data=1)  # constant growth


grid = UnitGrid([64, 64])  # generate grid
state = ScalarField.random_uniform(grid, 0.0, 0.5)  # generate initial condition

eq = CustomPDE()
result = eq.solve(state, dt=0.1, t_range=1e4)
result.plot(title=f"Total correction={eq.diagnostics['solver']['post_step_data']}")
