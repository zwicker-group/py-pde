'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np
import pytest

from ...fields import ScalarField
from ...grids import UnitGrid
from ...pdes import DiffusionPDE
from .. import Controller, ExplicitSolver, ImplicitSolver, ScipySolver
from .. import registered_solvers



def test_solver_registration():
    """ test solver registration """
    solvers = registered_solvers()
    assert 'explicit' in solvers
    assert 'scipy' in solvers



def test_solver_in_pde_class():
    """ test whether solver instances can be used in pde instances """
    field = ScalarField.random_uniform(UnitGrid([16, 16]), -1, 1)
    eq = DiffusionPDE()
    eq.solve(field, t_range=1, method=ScipySolver)



@pytest.mark.parametrize('solver_class', [ExplicitSolver, ImplicitSolver,
                                          ScipySolver])
def test_compare_solvers(solver_class):
    """ compare several solvers """
    field = ScalarField.random_uniform(UnitGrid([8, 8]), -1, 1)
    eq = DiffusionPDE()
    
    # ground truth
    c1 = Controller(ExplicitSolver(eq, scheme='runge-kutta'), t_range=.1,
                    tracker=None)
    s1 = c1.run(field, dt=5e-3)
    
    c2 = Controller(solver_class(eq), t_range=.1, tracker=None)
    with np.errstate(under='ignore'):
        s2 = c2.run(field, dt=5e-3)
                           
    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-2, atol=1e-2)

