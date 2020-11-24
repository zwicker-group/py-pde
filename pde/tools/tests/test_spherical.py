"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import math
import random

import numpy as np
import pytest
from pde.tools import spherical
from scipy import integrate, spatial


def test_volume_conversion():
    """ tests conversion of volume and radius of droplet """
    for dim in [1, 2, 3]:
        radius = 1 + random.random()
        volume = spherical.volume_from_radius(radius, dim=dim)
        radius2 = spherical.radius_from_volume(volume, dim=dim)
        assert radius2 == pytest.approx(radius)


def test_volume_conversion_numba():
    """ tests conversion of volume and radius of droplet using numba """
    for dim in [1, 2, 3]:
        r2v = spherical.make_volume_from_radius_compiled(dim)
        v2r = spherical.make_radius_from_volume_compiled(dim)
        radius = 1 + random.random()
        assert v2r(r2v(radius)) == pytest.approx(radius)


def test_surface():
    """ test whether the surface is calculated correctly """
    for dim in [1, 2, 3]:
        radius = 1 + random.random()
        eps = 1e-10
        vol1 = spherical.volume_from_radius(radius + eps, dim=dim)
        vol0 = spherical.volume_from_radius(radius, dim=dim)
        surface_approx = (vol1 - vol0) / eps
        surface = spherical.surface_from_radius(radius, dim=dim)
        assert surface == pytest.approx(surface_approx, rel=1e-3)

        r2s = spherical.make_surface_from_radius_compiled(dim)
        assert surface == pytest.approx(r2s(radius))

        if dim == 1:
            with pytest.raises(RuntimeError):
                spherical.radius_from_surface(surface, dim=dim)
        else:
            assert spherical.radius_from_surface(surface, dim=dim) == pytest.approx(
                radius
            )


def test_spherical_conversion():
    """ test the conversion between spherical and Cartesian coordinates """
    s2c = spherical.points_spherical_to_cartesian
    c2s = spherical.points_cartesian_to_spherical

    ps = np.random.randn(64, 3)
    np.testing.assert_allclose(s2c(c2s(ps)), ps)

    # enforce angles
    ps[:, 0] = np.abs(ps[:, 0])  # radius is positive
    ps[:, 1] %= np.pi  # θ is between 0 and pi
    ps[:, 2] %= 2 * np.pi  # φ is between 0 and 2 pi
    np.testing.assert_allclose(c2s(s2c(ps)), ps, rtol=1e-6)


def test_spherical_polygon_area():
    """ test the function get_spherical_polygon_area """
    area = spherical.get_spherical_polygon_area
    sector_area = area([[0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    assert sector_area == pytest.approx(np.pi / 2)


def test_spherical_voronoi():
    """ test spatial.SphericalVoronoi """
    # random points on the sphere
    ps = np.random.random((32, 3)) - 0.5
    ps /= np.linalg.norm(ps, axis=1)[:, None]

    voronoi = spatial.SphericalVoronoi(ps)
    voronoi.sort_vertices_of_regions()

    total = sum(
        spherical.get_spherical_polygon_area(voronoi.vertices[reg])
        for reg in voronoi.regions
    )
    assert total == pytest.approx(4 * np.pi)


@pytest.mark.parametrize("dim", range(1, 4))
def test_points_on_sphere(dim, tmp_path):
    """ test spatial.SphericalVoronoi """
    shell = spherical.PointsOnSphere.make_uniform(dim=dim)
    assert shell.dim == dim

    for balance_axes in [True, False]:
        ws = shell.get_area_weights(balance_axes=balance_axes)
        assert ws.sum() == pytest.approx(1)
        np.testing.assert_allclose(ws, 1 / len(shell.points), rtol=0.1)

    ws = shell.get_area_weights(balance_axes=True)
    np.testing.assert_allclose(ws @ shell.points, 0, atol=1e-15)

    path = tmp_path / f"test_points_on_sphere_{dim}.xyz"
    shell.write_to_xyz(path=path)
    assert path.stat().st_size > 0


def test_points_on_sphere_2():
    """ special tests for 2 dimensions """
    num = np.random.randint(3, 9)
    shell = spherical.PointsOnSphere.make_uniform(dim=2, num_points=num)
    assert num * shell.get_mean_separation() == pytest.approx(2 * np.pi)


def test_spherical_index():
    """ test the conversion of the spherical index """
    # check initial state
    assert spherical.spherical_index_lm(0) == (0, 0)
    assert spherical.spherical_index_k(0, 0) == 0

    # check conversion
    for k in range(20):
        l, m = spherical.spherical_index_lm(k)
        assert spherical.spherical_index_k(l, m) == k

    # check order
    k = 0
    for l in range(4):
        for m in range(-l, l + 1):
            assert spherical.spherical_index_k(l, m) == k
            k += 1

    for l in range(4):
        k_max = spherical.spherical_index_k(l, l)
        assert spherical.spherical_index_count(l) == k_max + 1

    for l in range(4):
        for m in range(-l, l + 1):
            is_optimal = m == l
            k = spherical.spherical_index_k(l, m)
            assert spherical.spherical_index_count_optimal(k + 1) == is_optimal


def test_spherical_harmonics_real():
    """ test spherical harmonics """
    # test real spherical harmonics for symmetric case
    for deg in range(4):
        for _ in range(5):
            θ = math.pi * random.random()
            φ = 2 * math.pi * random.random()
            y1 = spherical.spherical_harmonic_symmetric(deg, θ)
            y2 = spherical.spherical_harmonic_real(deg, 0, θ, φ)
            assert y1 == y2

    # test orthogonality of real spherical harmonics
    deg = 1
    Ylm = spherical.spherical_harmonic_real
    for m1 in range(-deg, deg + 1):
        for m2 in range(-deg, m1 + 1):

            def integrand(t, p):
                return Ylm(deg, m1, t, p) * Ylm(deg, m2, t, p) * np.sin(t)

            overlap = integrate.dblquad(
                integrand, 0, 2 * np.pi, lambda _: 0, lambda _: np.pi
            )[0]
            if m1 == m2:
                assert overlap == pytest.approx(1)
            else:
                assert overlap == pytest.approx(0)
