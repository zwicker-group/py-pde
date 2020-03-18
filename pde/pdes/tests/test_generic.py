'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from ... import pdes
from ...grids import UnitGrid
from ...fields import ScalarField



@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('pde_class', [pdes.KuramotoSivashinskyPDE, 
                                       pdes.KPZInterfacePDE,
                                       pdes.SwiftHohenbergPDE,
                                       pdes.DiffusionPDE,
                                       pdes.CahnHilliardPDE])
def test_pde_consistency(pde_class, dim):
    """ test some methods of generic PDE models """
    eq = pde_class()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)
    
    grid = UnitGrid([4] * dim)
    state = ScalarField.random_uniform(grid)
    field = eq.evolution_rate(state)
    assert field.grid == grid
    rhs = eq._make_pde_rhs_numba(state)
    np.testing.assert_allclose(field.data, rhs(state.data, 0))
    


def test_pde_consistency_test():
    """ test whether the consistency of a pde implementation is checked """

    class TestPDE(pdes.PDEBase):
        def evolution_rate(self, field, t=0):
            return 2 * field
        
        def _make_pde_rhs_numba(self, state):
            def impl(state_data, t):
                return 3 * state_data
            return impl
        
    eq = TestPDE()
    state = ScalarField.random_uniform(UnitGrid([4]))
    with pytest.raises(RuntimeError):
        eq.solve(state, t_range=5)
    