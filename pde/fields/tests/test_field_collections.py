"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import FieldCollection, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.fields.base import FieldBase
from pde.fields.tests.fixtures import iter_grids
from pde.tools.misc import skipUnlessModule


@pytest.mark.parametrize("grid", iter_grids())
def test_shapes_nfields(grid):
    """test single component field"""
    for num in [1, 3]:
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
    """test field collections"""
    grid = UnitGrid([3, 4])
    sf = ScalarField.random_uniform(grid, label="sf")
    vf = VectorField.random_uniform(grid, label="vf")
    tf = Tensor2Field.random_uniform(grid, label="tf")
    fields = FieldCollection([sf, vf, tf])
    assert fields.data.shape == (7, 3, 4)
    assert isinstance(str(fields), str)

    fields2 = FieldCollection.from_dict({"s": sf, "v": vf, "t": tf}, copy_fields=True)
    assert fields == fields2
    assert fields2.labels == ["s", "v", "t"]
    assert not np.shares_memory(fields[0].data, fields2[0].data)

    copy = fields[:2]
    assert isinstance(copy, FieldCollection)
    assert len(copy) == 2
    np.testing.assert_equal(copy[0].data, sf.data)
    assert not np.may_share_memory(copy[0].data, sf.data)
    np.testing.assert_equal(copy[1].data, vf.data)
    assert not np.may_share_memory(copy[1].data, vf.data)

    copy = fields[2:]
    assert isinstance(copy, FieldCollection)
    assert len(copy) == 1
    np.testing.assert_equal(copy[0].data, tf.data)
    assert not np.may_share_memory(copy[0].data, tf.data)

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
    assert all(np.allclose(i, 12) for i in fields.integrals)
    assert all(np.allclose(i, 1) for i in fields.averages)
    assert np.allclose(fields.magnitudes, np.sqrt([1, 2, 4]))

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
    """test copying data of collections"""
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
    """test field collections"""
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
    """test smoothing of a FieldCollection"""
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
    """test creating collections using scalar_random_uniform"""
    grid = UnitGrid([3, 4], periodic=[True, False])
    fc = FieldCollection.scalar_random_uniform(2, grid, label="c", labels=["a", "b"])
    assert fc.label == "c"
    assert fc[0].label == "a"
    assert fc[1].label == "b"
    assert fc[0].grid is grid
    assert fc[1].grid is grid
    assert not np.allclose(fc[0].data, fc[1].data)


def test_from_scalar_expressions():
    """test creating field collections from scalar expressions"""
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
@pytest.mark.interactive
def test_interactive_collection_plotting():
    """test the interactive plotting"""
    grid = UnitGrid([3, 3])
    sf = ScalarField.random_uniform(grid, 0.1, 0.9)
    vf = VectorField.random_uniform(grid, 0.1, 0.9)
    field = FieldCollection([sf, vf])
    field.plot_interactive(viewer_args={"show": False, "close": True})


def test_field_labels():
    """test the FieldCollection.labels property"""
    grid = UnitGrid([5])
    s1 = ScalarField(grid, label="s1")
    s2 = ScalarField(grid)
    fc = FieldCollection([s1, s2])

    assert fc.labels == ["s1", None]
    assert len(fc.labels) == 2
    assert fc.labels[0] == "s1"
    assert fc.labels.index("s1") == 0
    assert fc.labels.index(None) == 1
    with pytest.raises(ValueError):
        fc.labels.index("a")

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

    with pytest.raises(TypeError):
        fc.labels = [1, "b"]
    with pytest.raises(TypeError):
        fc.labels[0] = 1


def test_collection_1_field():
    """test field collections with only one field"""
    grid = UnitGrid([3])
    s1 = ScalarField(grid, label="a")
    fc = FieldCollection([s1])
    assert fc.labels == ["a"]

    fc.plot()


def test_collection_plotting():
    """test simple plotting of various fields on various grids"""
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


def test_from_data():
    """test the `from_data` method"""
    grid = UnitGrid([3, 5])
    s = ScalarField.random_uniform(grid, label="s1")
    v = VectorField.random_uniform(grid, label="v2")
    f1 = FieldCollection([s, v])

    f2 = FieldCollection.from_data(
        [ScalarField, VectorField],
        grid,
        data=f1.data,
        with_ghost_cells=False,
        labels=["a", "b"],
    )
    assert f2.labels == ["a", "b"]
    np.testing.assert_allclose(f1.data, f2.data)

    f3 = FieldCollection.from_data(
        [ScalarField, VectorField],
        grid,
        data=f1._data_full,
        with_ghost_cells=True,
        labels=["c", "d"],
    )
    assert f3.labels == ["c", "d"]
    np.testing.assert_allclose(f1.data, f2.data)
