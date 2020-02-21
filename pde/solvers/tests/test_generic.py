'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np

from ...fields import ScalarField
from ...grids import UnitGrid
from ...pdes import DiffusionPDE
from ...controller import Controller
from .. import ExplicitSolver, ImplicitSolver, ScipySolver
from ..base import SolverBase



def test_solver_registration():
    """ test solver registration """
    solvers = SolverBase.registered_solvers
    assert 'explicit' in solvers
    assert 'scipy' in solvers



def test_compare_solvers():
    """ compare several solvers """
    field = ScalarField.random_uniform(UnitGrid([16, 16]), -1, 1)
    eq = DiffusionPDE()
    
    # ground truth
    c1 = Controller(ExplicitSolver(eq, scheme='runge-kutta'), t_range=1,
                    tracker=None)
    s1 = c1.run(field, dt=1e-3)
    
    for solver_class in [ExplicitSolver, ImplicitSolver, ScipySolver]:
        c2 = Controller(solver_class(eq), t_range=1, tracker=None)
        with np.errstate(under='ignore'):
            s2 = c2.run(field, dt=1e-3)
                               
        np.testing.assert_allclose(s1.data, s2.data, rtol=1e-3, atol=1e-3)

