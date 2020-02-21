'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from .. import *
from ...grids import UnitGrid
from ...fields import ScalarField



@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('pde_class', [KuramotoSivashinskyPDE, KPZInterfacePDE,
                                       SwiftHohenbergPDE, DiffusionPDE,
                                       CahnHilliardPDE])
def test_pde_consistency(pde_class, dim):
    """ test some methods of the Kuramoto-Sivashinsky model """
    pde = pde_class()
    assert isinstance(str(pde), str)
    assert isinstance(repr(pde), str)
    
    grid = UnitGrid([4] * dim)
    state = ScalarField.random_uniform(grid)
    field = pde.evolution_rate(state)
    assert field.grid == grid
    rhs = pde._make_pde_rhs_numba(state)
    np.testing.assert_allclose(field.data, rhs(state.data, 0))
    
