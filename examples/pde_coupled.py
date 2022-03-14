r"""
Custom Class for coupled PDEs
=============================

This example shows how to solve a set of coupled PDEs, the
spatially coupled `FitzHugh–Nagumo model 
<https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model>`_, which is a simple model
for the excitable dynamics of coupled Neurons:

.. math::

    \partial_t u &= \nabla^2 u + u (u - \alpha) (1 - u) + w \\
    \partial_t w &= \epsilon u
    
Here, :math:`\alpha` denotes the external stimulus and :math:`\epsilon` defines
the recovery time scale. We implement this as a custom PDE class below.
"""

from pde import FieldCollection, PDEBase, UnitGrid


class FitzhughNagumoPDE(PDEBase):
    """FitzHugh–Nagumo model with diffusive coupling"""

    def __init__(self, stimulus=0.5, τ=10, a=0, b=0, bc="auto_periodic_neumann"):
        self.bc = bc
        self.stimulus = stimulus
        self.τ = τ
        self.a = a
        self.b = b

    def evolution_rate(self, state, t=0):
        v, w = state  # membrane potential and recovery variable

        v_t = v.laplace(bc=self.bc) + v - v**3 / 3 - w + self.stimulus
        w_t = (v + self.a - self.b * w) / self.τ

        return FieldCollection([v_t, w_t])


grid = UnitGrid([32, 32])
state = FieldCollection.scalar_random_uniform(2, grid)

eq = FitzhughNagumoPDE()
result = eq.solve(state, t_range=100, dt=0.01)
result.plot()
