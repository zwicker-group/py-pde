'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from ...fields import ScalarField
from ...grids import UnitGrid
from ...pdes import DiffusionPDE
from .. import Controller, ExplicitSolver



def test_compare_explicit():
    """ test explicit solvers """
    grid = UnitGrid([16, 16])
    field = ScalarField.random_uniform(grid, -1, 1)
    eq = DiffusionPDE()
    
    c1 = Controller(ExplicitSolver(eq), t_range=.1, tracker=None)
    s1 = c1.run(field, dt=2e-3)
    
    c2 = Controller(ExplicitSolver(eq, scheme='runge-kutta'), t_range=.1,
                    tracker=None)
    with np.errstate(under='ignore'):
        s2 = c2.run(field, dt=2e-3)
                               
    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-2, atol=1e-2)



@pytest.mark.parametrize('backend', ['numba', 'numpy'])
def test_stochastic_solvers(backend):
    """ test simple version of the stochastic solver """
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-6)
    
    solver1 = ExplicitSolver(eq, backend=backend)
    c1 = Controller(solver1, t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)
    
    solver2 = ExplicitSolver(seq, backend=backend)
    c2 = Controller(solver2, t_range=1, tracker=None)
    s2 = c2.run(field, dt=1e-3)
                               
    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-4, atol=1e-4)    
    assert not solver1.info['stochastic']
    assert solver2.info['stochastic']
    


def test_unsupported_stochastic_solvers():
    """ test some solvers that do not support stochasticity """
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1)
    eq = DiffusionPDE(noise=1)
    
    with pytest.raises(RuntimeError):
        eq.solve(field, 1, method='explicit', scheme='runge-kutta')
    with pytest.raises(RuntimeError):
        eq.solve(field, 1, method='scipy', scheme='runge-kutta')
