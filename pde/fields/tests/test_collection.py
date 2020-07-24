"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools

import numpy as np
import pytest

from ...grids import UnitGrid
from .. import FieldCollection, ScalarField, Tensor2Field, VectorField
from ..base import FieldBase
from .test_generic import iter_grids


def test_shapes_nfields():
    """ test single component field """
    for num, grid in itertools.product([1, 3], iter_grids()):
        fields = [ScalarField.random_uniform(grid) for _ in range(num)]
        field = FieldCollection(fields)
        data_shape = (num,) + grid.shape
        np.testing.assert_equal(field.data.shape, data_shape)
        for pf_single in field:
            np.testing.assert_equal(pf_single.data.shape, grid.shape)

        field_c = field.copy()
        np.testing.assert_allclose(field.data, field_c.data)
        assert field.grid == field_c.grid


def test_collections():
    """ test field collections """
    grid = UnitGrid([3, 4])
    sf = ScalarField.random_uniform(grid, label="sf")
    vf = VectorField.random_uniform(grid, label="vf")
    tf = Tensor2Field.random_uniform(grid, label="tf")
    fields = FieldCollection([sf, vf, tf])
    assert fields.data.shape == (7, 3, 4)
    assert isinstance(str(fields), str)

    fields.data[:] = 0
    np.testing.assert_allclose(sf.data, 0)
    np.testing.assert_allclose(vf.data, 0)
    np.testing.assert_allclose(tf.data, 0)

    assert fields[0] is fields["sf"]
    assert fields[1] is fields["vf"]
    assert fields[2] is fields["tf"]
    with pytest.raises(KeyError):
        fields["42"]

    sf.data = 1
    vf.data = 1
    tf.data = 1
    np.testing.assert_allclose(fields.data, 1)
    assert [np.allclose(i, 1) for i in fields.integrals]

    assert sf.data.shape == (3, 4)
    assert vf.data.shape == (2, 3, 4)
    assert tf.data.shape == (2, 2, 3, 4)

    c2 = FieldBase.from_state(fields.attributes, data=fields.data)
    assert c2 == fields
    assert c2.grid is grid

    attrs = FieldCollection.unserialize_attributes(fields.attributes_serialized)
    c2 = FieldCollection.from_state(attrs, data=fields.data)
    assert c2 == fields
    assert c2.grid is not grid

    fields["sf"] = 2.0
    np.testing.assert_allclose(sf.data, 2)
    with pytest.raises(KeyError):
        fields["42"] = 0


def test_collections_copy():
    """ test copying data of collections """
    grid = UnitGrid([2, 2])
    sf = ScalarField(grid, 0)
    vf = VectorField(grid, 1)
    fc = FieldCollection([sf, vf])

    data = np.r_[np.zeros(4), np.ones(8)]
    np.testing.assert_allclose(fc.data.flat, data)

    fc2 = fc.copy()
    assert fc.data is not fc2.data
    assert fc[0].data is not fc2[0].data
    assert fc[1].data is not fc2[1].data

    sf.data = 1
    np.testing.assert_allclose(fc.data.flat, np.ones(12))
    np.testing.assert_allclose(fc2.data.flat, data)


def test_collections_operators():
    """ test field collections """
    grid = UnitGrid([3, 4])
    sf = ScalarField(grid, 1)
    vf = VectorField(grid, 1)
    fields = FieldCollection([sf, vf])

    fields += fields
    np.testing.assert_allclose(fields.data, 2)
    np.testing.assert_allclose(sf.data, 2)
    np.testing.assert_allclose(vf.data, 2)

    fields = fields - 1
    np.testing.assert_allclose(fields.data, 1)

    fields = fields + fields
    np.testing.assert_allclose(fields.data, 2)

    fields *= 2
    np.testing.assert_allclose(fields.data, 4)


def test_smoothing_collection():
    """ test smoothing of a FieldCollection """
    grid = UnitGrid([3, 4], periodic=[True, False])
    sf = ScalarField.random_uniform(grid)
    vf = VectorField.random_uniform(grid)
    tf = Tensor2Field.random_uniform(grid)
    fields = FieldCollection([sf, vf, tf])
    sgm = 0.5 + np.random.random()

    out = fields.smooth(sigma=sgm)
    for i in range(3):
        np.testing.assert_allclose(out[i].data, fields[i].smooth(sgm).data)

    out.data = 0
    fields.smooth(sigma=sgm, out=out)
    for i in range(3):
        np.testing.assert_allclose(out[i].data, fields[i].smooth(sgm).data)


def test_scalar_random_uniform():
    """ test creating collections using scalar_random_uniform """
    grid = UnitGrid([3, 4], periodic=[True, False])
    fc = FieldCollection.scalar_random_uniform(2, grid, label="c", labels=["a", "b"])
    assert fc.label == "c"
    assert fc[0].label == "a"
    assert fc[1].label == "b"
    assert fc[0].grid is grid
    assert fc[1].grid is grid
    assert not np.allclose(fc[0].data, fc[1].data)


def test_from_scalar_expressions():
    """ test creating field collections from scalar expressions """
    grid = UnitGrid([3])
    expressions = ["x**2", "1"]
    fc = FieldCollection.from_scalar_expressions(
        grid, expressions=expressions, label="c", labels=["a", "b"]
    )
    assert fc.label == "c"
    assert fc[0].label == "a"
    assert fc[1].label == "b"
    assert fc[0].grid is grid
    assert fc[1].grid is grid
    np.testing.assert_allclose(fc[0].data, (np.arange(3) + 0.5) ** 2)
    np.testing.assert_allclose(fc[1].data, 1)
