r"""
Custom PDE class: SIR model
===========================

This example implements a `spatially coupled SIR model 
<https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_ with 
the following dynamics for the density of susceptible, infected, and recovered
individuals:

.. math::

    \partial_t s &= D \nabla^2 s - \beta is \\
    \partial_t i &= D \nabla^2 i + \beta is - \gamma i \\
    \partial_t r &= D \nabla^2 r + \gamma i
    
Here, :math:`D` is the diffusivity, :math:`\beta` the infection rate, and
:math:`\gamma` the recovery rate.
"""

from pde import UnitGrid, ScalarField, FieldCollection, PDEBase, PlotTracker


class SIRPDE(PDEBase):
    """ SIR-model with diffusive mobility """
    
    def __init__(self, beta=0.3, gamma=0.9, diffusivity=0.1, bc='natural'):
        self.beta = beta  # transmission rate
        self.gamma = gamma  # recovery rate
        self.diffusivity = diffusivity  # spatial mobility
        self.bc = bc  # boundary condition
        
    def get_state(self, s, i):
        """ generate a suitable initial state"""
        norm = (s + i).data.max()  # maximal density
        if norm > 1:
            s /= norm
            i /= norm
        s.label = 'Susceptible'
        i.label = 'Infected'
            
        # create recovered field
        r = ScalarField(s.grid, data=1 - s - i, label='Recovered')
        return FieldCollection([s, i, r])
        
    def evolution_rate(self, state, t=0):
        s, i, r = state
        diff = self.diffusivity
        ds_dt = diff * s.laplace(self.bc) - self.beta * i * s
        di_dt = diff * i.laplace(self.bc) + self.beta * i * s - self.gamma * i
        dr_dt = diff * r.laplace(self.bc) + self.gamma * i
        return FieldCollection([ds_dt, di_dt, dr_dt])


eq = SIRPDE(beta=2, gamma=0.1)

# initialize state
grid = UnitGrid([32, 32])
s = ScalarField(grid, 1)
i = ScalarField(grid, 0)
i.data[0, 0] = 1
state = eq.get_state(s, i)

# simulate the pde
tracker = PlotTracker(interval=10, plot_arguments={'vmin': 0, 'vmax': 1})
sol = eq.solve(state, t_range=50, dt=1e-2, tracker=['progress', tracker])