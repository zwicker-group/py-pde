'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import itertools
import pytest

from ... import UnitGrid
from ..local import BCBase
from ..axis import BoundaryPair, get_boundary_axis



def test_boundary_pair():
    """ test setting boundary conditions for whole axis """
    g = UnitGrid([2, 3])
    b = ['value', {'type': 'derivative', 'value': 1}]
    for bl, bh in itertools.product(b, b):
        bc = BoundaryPair.from_data(g, 0, [bl, bh])
        blo = BCBase.from_data(g, 0, upper=False, data=bl)
        bho = BCBase.from_data(g, 0, upper=True, data=bh)
        
        assert bc.low == blo
        assert bc.high == bho
        assert bc == BoundaryPair(blo, bho)
        if bl == bh:
            assert bc == BoundaryPair.from_data(g, 0, bl)
                      
        assert bc.check_value_rank(0) is None
        if bl == bh == 'value':
            assert bc.check_value_rank(1) is None
        else:
            with pytest.raises(RuntimeError):
                bc.check_value_rank(1)
                            
                            
                            
def test_get_axis_boundaries():
    """ test setting boundary conditions including periodic ones """
    g = UnitGrid([2])
    for data in ['value', 'derivative', 'periodic']:
        b = get_boundary_axis(g, 0, data)
        assert str(b) == '"' + data + '"'
        
        if data == 'periodic':
            assert b.periodic
        else:
            assert not b.periodic
