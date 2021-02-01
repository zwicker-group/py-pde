"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import FieldCollection, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.fields.base import FieldBase
from pde.tools.misc import skipUnlessModule


def test_shapes_nfields(example_grid):
    """ test single component field """
    for num in [1, 3]:
        fields = [ScalarField.random_uniform(example_grid) for _ in range(num)]
        field = FieldCollection(fields)
        data_shape = (num,) + example_grid.shape
        np.testing.assert_equal(field.data.shape, data_shape)
        for pf_single in field:
            np.testing.assert_equal(pf_single.data.shape, example_grid.shape)

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

    fields.plot(subplot_args=[{}, {"scale": 1}, {"colorbar": False}])


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

    # special case
    fc = FieldCollection([sf, sf])
    fc[0] = 2
    np.testing.assert_allclose(fc[0].data, 2)
    np.testing.assert_allclose(fc[1].data, 1)


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


@skipUnlessModule("napari")
def test_interactive_collection_plotting():
    """ test the interactive plotting """
    grid = UnitGrid([3, 3])
    sf = ScalarField.random_uniform(grid, 0.1, 0.9)
    vf = VectorField.random_uniform(grid, 0.1, 0.9)
    field = FieldCollection([sf, vf])
    field.plot_interactive(viewer_args={"show": False, "close": True})


def test_field_labels():
    """ test the FieldCollection.labels property """
    grid = UnitGrid([5])
    s1 = ScalarField(grid, label="s1")
    s2 = ScalarField(grid)
    fc = FieldCollection([s1, s2])

    assert fc.labels == ["s1", None]
    assert len(fc.labels) == 2
    assert fc.labels[0] == "s1"
    fc.labels = ["a", "b"]
    assert fc.labels == ["a", "b"]
    fc.labels[0] = "c"
    assert fc.labels == ["c", "b"]
    assert str(fc.labels) == str(["c", "b"])
    assert repr(fc.labels) == repr(["c", "b"])

    assert fc.labels[0:1] == ["c"]
    assert fc.labels[:] == ["c", "b"]
    fc.labels[0:1] = "d"
    assert fc.labels == ["d", "b"]

    fc.labels[:] = "a"
    assert fc.labels == ["a", "a"]

    labels = fc.labels[:]
    labels[0] = "e"
    assert fc.labels == ["a", "a"]

    fc = FieldCollection([s1, s2], labels=[None, "b"])
    assert fc.labels == [None, "b"]
    fc = FieldCollection([s1, s2], labels=["a", "b"])
    assert fc.labels == ["a", "b"]


def test_collection_1_field():
    """ test field collections with only one field """
    grid = UnitGrid([3])
    s1 = ScalarField(grid, label="a")
    fc = FieldCollection([s1])
    assert fc.labels == ["a"]

    fc.plot()


def test_collection_plotting():
    """ test simple plotting of various fields on various grids """
    grid = UnitGrid([5])
    s1 = ScalarField(grid, label="s1")
    s2 = ScalarField(grid)
    fc = FieldCollection([s1, s2])

    # test setting different figure sizes
    fc.plot(figsize="default")
    fc.plot(figsize="auto")
    fc.plot(figsize=(3, 3))

    # test different arrangements
    fc.plot(arrangement="horizontal")
    fc.plot(arrangement="vertical")
