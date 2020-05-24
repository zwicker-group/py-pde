'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from .. import PDE, SwiftHohenbergPDE
from ...grids import UnitGrid
from ...fields import ScalarField, VectorField, FieldCollection
from ...grids.tests.test_generic import iter_grids



def test_pde_wrong_input():
    """ test some wrong input """
    with pytest.raises(RuntimeError):
        PDE({'t': 1})
        
    grid = UnitGrid([4])
    eq = PDE({'u': 1})
    assert eq.expressions == {'u': '1.0'}
    with pytest.raises(ValueError):
        eq.evolution_rate(FieldCollection.scalar_random_uniform(2, grid))
    
    eq = PDE({'u': 1, 'v': 2})
    assert eq.expressions == {'u': '1.0', 'v': '2.0'}
    with pytest.raises(ValueError):
        eq.evolution_rate(ScalarField.random_uniform(grid))
        


def test_pde_scalar():
    """ test PDE with a single scalar field """
    eq = PDE({'u': 'laplace(u) + exp(-t) + sin(t)'})
    grid = UnitGrid([8])
    field = ScalarField.random_normal(grid)
    
    res_a = eq.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend='numba', tracker=None)
    
    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)
    
    

def test_pde_vector():
    """ test PDE with a single vector field """
    eq = PDE({'u': 'vector_laplace(u) + exp(-t)'})
    grid = UnitGrid([8, 8])
    field = VectorField.random_normal(grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend='numba', tracker=None)
    
    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)    
    
    
    
def test_pde_2scalar():
    """ test PDE with two scalar fields """
    eq = PDE({'u': 'laplace(u) - u', 'v': '- u * v'})
    grid = UnitGrid([8])
    field = FieldCollection.scalar_random_uniform(2, grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend='numba', tracker=None)
    
    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)
    
    
    
def test_pde_vector_scalar():
    """ test PDE with a vector and a scalar field """
    eq = PDE({'u': 'vector_laplace(u) - u + gradient(v)',
              'v': '- divergence(u)'})
    grid = UnitGrid([8, 8])
    field = FieldCollection([VectorField.random_uniform(grid),
                             ScalarField.random_uniform(grid)])

    res_a = eq.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend='numba', tracker=None)
    
    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)  



@pytest.mark.parametrize('grid', iter_grids())
def test_compare_swift_hohenberg(grid):
    """ compare custom class to swift-Hohenberg """
    rate, kc2, delta = np.random.uniform(0.5, 2, size=3)
    eq1 = SwiftHohenbergPDE(rate=rate, kc2=kc2, delta=delta)
    eq2 = PDE({'u': f"({rate} - {kc2}**2) * u - 2 * {kc2} * laplace(u) "
                    f"- laplace(laplace(u)) + {delta} * u**2 - u**3"})

    field = ScalarField.random_uniform(grid)
    res1 = eq1.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    res2 = eq2.solve(field, t_range=1, dt=0.01, backend='numpy', tracker=None)
    
    res1.assert_field_compatible(res1)
    np.testing.assert_allclose(res1.data, res2.data)
    
    
    
def test_custom_operators():
    """ test using a custom operator """
    grid = UnitGrid([32])
    field = ScalarField.random_normal(grid)
    eq = PDE({'u': 'undefined(u)'})
    
    with pytest.raises(ValueError):
        eq.evolution_rate(field)
        
    
    def make_op(state):
        return lambda state: state
    UnitGrid.register_operator('undefined', make_op)
    
    eq._cache = {}  # reset cache
    res = eq.evolution_rate(field)
    np.testing.assert_allclose(field.data, res.data)
    
    del UnitGrid._operators['undefined']  # reset original state
    
    
    
@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_pde_noise(backend):
    """ test noise operator on PDE class """
    grid = UnitGrid([64, 64])
    state = FieldCollection([ScalarField(grid), ScalarField(grid)])
    
    eq = PDE({'a': 0, 'b': 0}, noise=.5)
    res = eq.solve(state, t_range=1, backend=backend, dt=1)
    assert res.data.std() == pytest.approx(.5, rel=0.1)

    eq = PDE({'a': 0, 'b': 0}, noise=[0.01, 2.])
    res = eq.solve(state, t_range=1, backend=backend, dt=1)
    assert res.data[0].std() == pytest.approx(0.01, rel=0.1)
    assert res.data[1].std() == pytest.approx(2., rel=0.1)

    with pytest.raises(RuntimeError):
        eq = PDE({'a': 0}, noise=[0.01, 2.])
        eq.solve(ScalarField(grid), t_range=1, backend=backend, dt=1)
    