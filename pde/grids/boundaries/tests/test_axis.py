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
        assert list(bc) == [blo, bho]
                      
        bc.check_value_rank(0)
        with pytest.raises(RuntimeError):
            bc.check_value_rank(1)
                
    data = {'low': {'value': 1}, 'high': {'derivative': 2}}
    bc1 = BoundaryPair.from_data(g, 0, data)
    bc2 = BoundaryPair.from_data(g, 0, data)
    assert bc1 == bc2 and bc1 is not bc2
    bc2 = BoundaryPair.from_data(g, 1, data)
    assert bc1 != bc2 and bc1 is not bc2
    
    # miscellaneous methods
    bc1.set_value(0)
    b_lo, b_hi = bc1
    assert b_lo == BCBase.from_data(g, 0, False, {'value': 0})
    assert b_hi == BCBase.from_data(g, 0, True, {'derivative': 0})
    
                            
                            
def test_get_axis_boundaries():
    """ test setting boundary conditions including periodic ones """
    g = UnitGrid([2])
    for data in ['value', 'derivative', 'periodic']:
        b = get_boundary_axis(g, 0, data)
        assert str(b) == '"' + data + '"'
        
        if data == 'periodic':
            assert b.periodic
            assert len(list(b)) == 0
        else:
            assert not b.periodic
            assert len(list(b)) == 2
            