"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, PolarSymGrid, ScalarField, Tensor2Field, UnitGrid
from pde.fields.base import FieldBase
from pde.fields.tests.fixtures import iter_grids


def test_tensors_basic():
    """test some tensor calculations"""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])

    t1 = Tensor2Field(grid, np.full((2, 2) + grid.shape, 1))
    t2 = Tensor2Field(grid, np.full((2, 2) + grid.shape, 2))
    np.testing.assert_allclose(t2.average, [[2, 2], [2, 2]])
    assert t1.magnitude == pytest.approx(2)

    assert t1["x", "x"] == t1[0, 0]
    assert t1["x", 1] == t1[0, "y"] == t1[0, 1]
    t1[0, 0] = t1[0, 0]

    t3 = t1 + t2
    assert t3.grid == grid
    np.testing.assert_allclose(t3.data, 3)
    t1 += t2
    np.testing.assert_allclose(t1.data, 3)

    field = Tensor2Field.random_uniform(grid)
    trace = field.trace()

    assert isinstance(trace, ScalarField)
    np.testing.assert_allclose(trace.data, field.data.trace())

    t1 = Tensor2Field(grid)
    t1[0, 0] = 1
    t1[0, 1] = 2
    t1[1, 0] = 3
    t1[1, 1] = 4
    for method, value in [
        ("min", 1),
        ("max", 4),
        ("norm", np.linalg.norm([[1, 2], [3, 4]])),
        ("squared_sum", 30),
        ("norm_squared", 30),
        ("trace", 5),
        ("invariant1", 5),
        ("invariant2", -1),
    ]:
        p1 = t1.to_scalar(method)
        assert p1.data.shape == grid.shape
        np.testing.assert_allclose(p1.data, value)

    for idx in ((1,), (1, 2, 3), (1.5, 2), ("a", "b"), 1.0):
        with pytest.raises(IndexError):
            t1[idx]

    t2 = FieldBase.from_state(t1.attributes, data=t1.data)
    assert t1 == t2
    assert t1.grid is t2.grid

    attrs = Tensor2Field.unserialize_attributes(t1.attributes_serialized)
    t2 = FieldBase.from_state(attrs, data=t1.data)
    assert t1 == t2
    assert t1.grid is not t2.grid


@pytest.mark.parametrize("grid", [UnitGrid([1, 1]), PolarSymGrid(2, 1)])
def test_tensors_transpose(grid):
    """test transposing tensors"""

    def broadcast(arr):
        return np.asarray(arr)[(...,) + (np.newaxis,) * grid.num_axes]

    field = Tensor2Field(grid, broadcast([[0, 1], [2, 3]]))
    field_T = field.transpose(label="altered")
    assert field_T.label == "altered"
    np.testing.assert_allclose(field_T.data, broadcast([[0, 2], [1, 3]]))


def test_tensor_symmetrize():
    """test advanced tensor calculations"""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [2, 2])
    t1 = Tensor2Field(grid)
    t1.data[0, 0, :] = 1
    t1.data[0, 1, :] = 2
    t1.data[1, 0, :] = 3
    t1.data[1, 1, :] = 4

    # traceless = False
    t2 = t1.copy()
    t1.symmetrize(make_traceless=False, inplace=True)
    tr = t1.trace()
    assert np.all(tr.data == 5)
    t1_trans = np.swapaxes(t1.data, 0, 1)
    np.testing.assert_allclose(t1.data, t1_trans.data)

    ts = t1.copy()
    ts.symmetrize(make_traceless=False, inplace=True)
    np.testing.assert_allclose(t1.data, ts.data)

    # traceless = True
    t2.symmetrize(make_traceless=True, inplace=True)
    tr = t2.trace()
    assert np.all(tr.data == 0)
    t2_trans = np.swapaxes(t2.data, 0, 1)
    np.testing.assert_allclose(t2.data, t2_trans.data)

    ts = t2.copy()
    ts.symmetrize(make_traceless=True, inplace=True)
    np.testing.assert_allclose(t2.data, ts.data)


@pytest.mark.parametrize("grid", iter_grids())
def test_insert_tensor(grid):
    """test the `insert` method"""
    f = Tensor2Field(grid)
    a = np.random.random(f.data_shape)

    c = tuple(grid.get_random_point(coords="cell"))
    c_data = (Ellipsis,) + c
    p = grid.transform(c, "cell", "grid")
    f.insert(p, a)
    np.testing.assert_almost_equal(f.data[c_data], a / grid.cell_volumes[c])

    f.insert(grid.get_random_point(coords="grid"), a)
    np.testing.assert_almost_equal(f.integral, 2 * a)

    f.data = 0  # reset
    insert = grid.make_inserter_compiled()
    c = tuple(grid.get_random_point(coords="cell"))
    c_data = (Ellipsis,) + c
    p = grid.transform(c, "cell", "grid")
    insert(f.data, p, a)
    np.testing.assert_almost_equal(f.data[c_data], a / grid.cell_volumes[c])

    insert(f.data, grid.get_random_point(coords="grid"), a)
    np.testing.assert_almost_equal(f.integral, 2 * a)


def test_tensor_invariants():
    """test the invariants"""
    # dim == 1
    f = Tensor2Field.random_uniform(UnitGrid([3]))
    np.testing.assert_allclose(
        f.to_scalar("invariant1").data, f.to_scalar("invariant3").data
    )
    np.testing.assert_allclose(f.to_scalar("invariant2").data, 0)

    # dim == 2
    f = Tensor2Field.random_uniform(UnitGrid([3, 3]))
    invs = [f.to_scalar(f"invariant{i}").data for i in range(1, 4)]
    np.testing.assert_allclose(2 * invs[1], invs[2])

    a = np.random.uniform(0, 2 * np.pi)  # pick random rotation angle
    rot = Tensor2Field(f.grid)
    rot.data[0, 0, ...] = np.cos(a)
    rot.data[0, 1, ...] = np.sin(a)
    rot.data[1, 0, ...] = -np.sin(a)
    rot.data[1, 1, ...] = np.cos(a)
    f_rot = rot @ f @ rot.transpose()  # apply the transpose

    for i, inv in enumerate(invs, 1):
        np.testing.assert_allclose(
            inv,
            f_rot.to_scalar(f"invariant{i}").data,
            err_msg=f"Mismatch in invariant {i}",
        )

    # dim == 3
    from scipy.spatial.transform import Rotation

    f = Tensor2Field.random_uniform(UnitGrid([1, 1, 1]))
    rot = Tensor2Field(f.grid)
    rot_mat = Rotation.from_rotvec(np.random.randn(3)).as_matrix()
    rot.data = rot_mat.reshape(3, 3, 1, 1, 1)
    f_rot = rot @ f @ rot.transpose()  # apply the transpose
    for i in range(1, 4):
        np.testing.assert_allclose(
            f.to_scalar(f"invariant{i}").data,
            f_rot.to_scalar(f"invariant{i}").data,
            err_msg=f"Mismatch in invariant {i}",
        )


def test_complex_tensors():
    """test some complex tensor fields"""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    shape = (2, 2, 2) + grid.shape
    numbers = np.random.random(shape) + np.random.random(shape) * 1j
    t1 = Tensor2Field(grid, numbers[0])
    t2 = Tensor2Field(grid, numbers[1])
    assert t1.is_complex and t2.is_complex

    for backend in ["numba", "numpy"]:
        dot_op = t1.make_dot_operator(backend)

        # test dot product
        res = dot_op(t1.data, t2.data)
        for t in (t1 @ t2, t1.dot(t2)):
            assert isinstance(t, Tensor2Field)
            assert t.grid is grid
            np.testing.assert_allclose(t.data, res)

        # test without conjugate
        dot_op = t1.make_dot_operator(backend, conjugate=False)
        res = t1.dot(t2, conjugate=False)
        np.testing.assert_allclose(dot_op(t1.data, t2.data), res.data)


def test_from_expressions():
    """test initializing tensor fields with expressions"""
    grid = UnitGrid([4, 4])
    tf = Tensor2Field.from_expression(grid, [[1, 1], ["x**2", "x * y"]])
    xs = grid.cell_coords[..., 0]
    ys = grid.cell_coords[..., 1]
    np.testing.assert_allclose(tf.data[0, 1], 1)
    np.testing.assert_allclose(tf.data[0, 1], 1)
    np.testing.assert_allclose(tf.data[1, 0], xs**2)
    np.testing.assert_allclose(tf.data[1, 1], xs * ys)

    # corner case
    with pytest.raises(ValueError):
        Tensor2Field.from_expression(grid, "xy")
    with pytest.raises(ValueError):
        Tensor2Field.from_expression(grid, ["xy"])
    with pytest.raises(ValueError):
        Tensor2Field.from_expression(grid, ["x"] * 3)
    with pytest.raises(ValueError):
        Tensor2Field.from_expression(grid, [["x"], [1, 1]])
