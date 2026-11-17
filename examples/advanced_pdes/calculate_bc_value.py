r"""
Calculate value of boundary condition
=====================================

This example demonstrates low-level access to boundary conditions. In this case, the
value at the boundary condition is calculated on the fly using the boundary condition
:class:`~pde.grids.boundaries.local.UserValueBC`.
"""

from typing import NamedTuple

import numpy as np

from pde import PDEBase, ScalarField, UnitGrid


class BCValueArgs(NamedTuple):
    """Container for the arguments passed to the boundary condition"""

    value: np.ndarray
    t: float


class MyPDE(PDEBase):
    """A custom PDE, which implements a diffusion PDE with custom BCs."""

    def __init__(self, bc):
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """Evaluate the right hand side of the evolution equation."""
        bc_values = np.ones(state.data.shape[0])  # custom BC calculation
        args = BCValueArgs(value=bc_values, t=t)
        return state.laplace(bc=self.bc, args=args)

    def make_evolution_rate(self, state, backend):
        """Create compiled implementation of right hand side"""
        laplace = state.grid.make_operator("laplace", bc=self.bc, backend=backend)

        def pde_rhs(state_data, t=0):
            """Compilable implementation of right hand side."""
            bc_values = np.ones(state_data.shape[0])  # custom BC calculation
            args = BCValueArgs(value=bc_values, t=t)
            return laplace(state_data, args=args)

        return pde_rhs


# define PDE and initial state
eq = MyPDE(bc={"*": "value", "y+": "user_value"})
field = ScalarField.random_normal(UnitGrid([32, 32]))

# solve the PDE with custom boundary conditions
res = eq.solve(field, t_range=10, dt=0.01, backend="numpy")
res.plot()
