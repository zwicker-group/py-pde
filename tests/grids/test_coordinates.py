"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pickle

import numpy as np
import pytest

from pde.grids import coordinates


def iter_coordinates():
    """generator providing some test coordinate systems"""
    yield coordinates.CartesianCoordinates(1)
    yield coordinates.CartesianCoordinates(2)
    yield coordinates.CartesianCoordinates(3)
    yield coordinates.PolarCoordinates()
    yield coordinates.SphericalCoordinates()
    yield coordinates.CylindricalCoordinates()
    yield coordinates.BipolarCoordinates()
    yield coordinates.BipolarCoordinates(3)
    yield coordinates.BisphericalCoordinates()
    yield coordinates.BisphericalCoordinates(3)


@pytest.mark.parametrize("c", iter_coordinates())
def test_basic_coordinates(c, rng):
    """test basic coordinate properties"""
    assert len(c.coordinate_limits) == c.dim
    assert len(c.axes) == c.dim
    x = rng.uniform(size=c.dim)
    p = c.pos_from_cart(x)
    np.testing.assert_allclose(c.pos_to_cart(p), x)

    assert pickle.loads(pickle.dumps(c)) == c


@pytest.mark.parametrize("c", iter_coordinates())
def test_coordinate_volume_factors(c, rng):
    """test basic coordinate properties"""
    p1 = c.pos_from_cart(rng.uniform(-1, 1, size=c.dim))
    p2 = c.pos_from_cart(rng.uniform(-1, 1, size=c.dim))
    p_l = np.minimum(p1, p2)
    p_h = np.maximum(p1, p2)
    vol1 = c.cell_volume(p_l, p_h)
    vol2 = coordinates.CoordinatesBase._cell_volume(c, p_l, p_h)
    assert vol2 == pytest.approx(vol1)


@pytest.mark.parametrize("c", iter_coordinates())
def test_coordinate_metric(c, rng):
    x = rng.uniform(size=c.dim)
    p = c.pos_from_cart(x)

    # test mapping Jacobian
    J1 = coordinates.CoordinatesBase._mapping_jacobian(c, p)
    J2 = c.mapping_jacobian(p)
    np.testing.assert_almost_equal(J1, J2)

    # test volume element
    v1 = coordinates.CoordinatesBase._volume_factor(c, p)
    v2 = c.volume_factor(p)
    assert v1 == pytest.approx(v2)
    assert v2 == pytest.approx(np.linalg.det(J2))

    g = c.metric(p)
    det_g = np.linalg.det(g)
    det_J = np.linalg.det(J2)
    assert det_g == pytest.approx(det_J**2)


@pytest.mark.parametrize("c", iter_coordinates())
def test_coordinate_vector_fields(c, rng):
    """test basic coordinate properties"""
    # anchor point
    x1 = rng.uniform(-1, 1, size=c.dim)
    p = c.pos_from_cart(x1)

    # rotation must be orthogonal matrices
    rot = c.basis_rotation(p)
    assert np.linalg.det(rot) == pytest.approx(1)
    np.testing.assert_allclose(rot @ rot.T, np.eye(c.dim), atol=1e-16)

    # vector components
    for i in range(c.dim):
        v = np.eye(c.dim)[i]

        eps = 1e-8
        x2 = c.pos_to_cart(p + eps * v)  # slightly moved point
        dx = (x2 - x1) / eps
        dx /= np.linalg.norm(dx)
        np.testing.assert_allclose(dx, c.vec_to_cart(p, v), atol=1e-6)


def test_invalid_coordinates():
    """test some invalid initializations"""
    with pytest.raises(ValueError):
        coordinates.CartesianCoordinates(0)
    with pytest.raises(ValueError):
        coordinates.CartesianCoordinates(-1)
    with pytest.raises(ValueError):
        coordinates.BipolarCoordinates(0)
    with pytest.raises(ValueError):
        coordinates.BipolarCoordinates(-1)
    with pytest.raises(ValueError):
        coordinates.BisphericalCoordinates(0)
    with pytest.raises(ValueError):
        coordinates.BisphericalCoordinates(-1)
