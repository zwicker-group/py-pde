"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.cuboid import Cuboid, asanyarray_flags


def test_cuboid_2d():
    """test Cuboid class in 2d"""
    c = Cuboid([-1, -1], [2, 2])
    assert c.dim == 2
    assert c.volume == 4
    assert c.diagonal == pytest.approx(np.sqrt(8))
    np.testing.assert_array_equal(c.centroid, [0, 0])

    c = Cuboid([0, 0], [1, -1])
    assert c.diagonal == pytest.approx(np.sqrt(2))
    np.testing.assert_array_equal(c.pos, [0, -1])
    np.testing.assert_array_equal(c.size, [1, 1])

    c = Cuboid.from_points([1, -1], [0, 0])
    assert c.diagonal == pytest.approx(np.sqrt(2))
    np.testing.assert_array_equal(c.pos, [0, -1])
    np.testing.assert_array_equal(c.size, [1, 1])

    c = Cuboid.from_centerpoint([0, 0], [2, -2])
    assert c.diagonal == pytest.approx(np.sqrt(8))
    np.testing.assert_array_equal(c.pos, [-1, -1])
    np.testing.assert_array_equal(c.size, [2, 2])

    c = Cuboid.from_points([0, 1], [2, 3])
    verts = set(c.vertices)
    assert len(verts) == 4
    assert (0, 1) in verts
    assert (2, 3) in verts
    assert (0, 1) in verts
    assert (2, 3) in verts

    c = Cuboid([-1, -1], [2, 2])
    assert c.diagonal == pytest.approx(np.sqrt(8))

    c1 = c.buffer(1)
    assert c1.volume == 16

    c = Cuboid([0, 2], [2, 4])
    c.centroid = [0, 0]
    np.testing.assert_array_equal(c.pos, [-1, -2])
    np.testing.assert_array_equal(c.size, [2, 4])

    c = Cuboid([0, 0], [2, 2])
    np.testing.assert_array_equal(c.contains_point([]), [])
    np.testing.assert_array_equal(c.contains_point([1, 1]), [True])
    np.testing.assert_array_equal(c.contains_point([3, 3]), [False])
    np.testing.assert_array_equal(c.contains_point([[1, 1], [3, 3]]), [True, False])
    np.testing.assert_array_equal(c.contains_point([[1, 3], [3, 1]]), [False, False])
    np.testing.assert_array_equal(c.contains_point([[1, -1], [-1, 1]]), [False, False])

    with pytest.raises(ValueError):
        c.mutable = False
        c.centroid = [0, 0]

    # test surface area
    c = Cuboid([0, 0], [1, 3])
    assert c.surface_area == 8
    c = Cuboid([0, 0], [1, 0])
    assert c.surface_area == 2
    c = Cuboid([0, 0], [0, 0])
    assert c.surface_area == 0


def test_cuboid_add():
    """test adding two cuboids"""
    assert Cuboid([1], [2]) + Cuboid([1], [2]) == Cuboid([1], [2])
    assert Cuboid([1], [2]) + Cuboid([0], [1]) == Cuboid([0], [3])
    assert Cuboid([1], [2]) + Cuboid([2], [2]) == Cuboid([1], [3])
    with pytest.raises(RuntimeError):
        Cuboid([1], [2]) + Cuboid([1, 2], [1, 1])


def test_cuboid_nd():
    """test Cuboid class in n dimensions"""
    dim = np.random.randint(5, 10)
    size = np.random.randn(dim)
    c = Cuboid(np.random.randn(dim), size)
    assert c.dim == dim
    assert c.diagonal == pytest.approx(np.linalg.norm(size))
    c2 = Cuboid.from_bounds(c.bounds)
    np.testing.assert_allclose(c.bounds, c2.bounds)

    # test surface area
    c = Cuboid([0], [1])
    assert c.surface_area == 2
    c = Cuboid([0, 0, 0], [1, 2, 3])
    assert c.surface_area == 22

    for n in range(1, 5):
        c = Cuboid(np.zeros(n), np.full(n, 3))
        assert c.surface_area == 2 * n * 3 ** (n - 1)


def test_asanyarray_flags():
    """test the asanyarray_flags function"""
    assert np.arange(3) is not asanyarray_flags(range(3))

    a = np.random.random(3).astype(np.double)
    assert a is asanyarray_flags(a)
    assert a is asanyarray_flags(a, np.double)
    assert a is asanyarray_flags(a, writeable=True)
    assert a is not asanyarray_flags(a, np.intc)
    assert a is not asanyarray_flags(a, writeable=False)

    for dtype in (np.intc, np.double):
        b = asanyarray_flags(a, dtype)
        assert b.dtype == dtype

    for writeable in (True, False):
        b = asanyarray_flags(a, writeable=writeable)
        assert b.flags.writeable == writeable
