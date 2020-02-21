'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from ..grids import CartesianGrid
from ..fields import ScalarField
from ..analysis import get_length_scale



def test_get_length_scale():
    """ test getting length scales """
    grid = CartesianGrid([[0, 20 * np.pi]], 256, periodic=True)
    sf = ScalarField.from_expression(grid, 'sin(x)')
    for method in ['structure_factor_mean', 'structure_factor_peak']:
        res = get_length_scale(sf, method=method)
        assert res == pytest.approx(2 * np.pi)
