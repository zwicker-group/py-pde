r"""
Module collecting functions for handling spherical geometry.

The coordinate systems use the following convention for polar coordinates
:math:`(r, \phi)`, where :math:`r` is the radial coordinate and :math:`\phi` is
the polar angle:

.. math::
    \begin{cases}
        x = r \cos(\phi) &\\
        y = r \sin(\phi) &
    \end{cases}
    \text{for} \; r \in [0, \infty] \;
    \text{and} \; \phi \in [0, 2\pi)
    
Similarly, for spherical coordinates :math:`(r, \theta, \phi)`, where :math:`r`
is the radial coordinate, :math:`\theta` is the azimuthal angle, and
:math:`\phi` is the polar angle, we use

.. math::
    \begin{cases}
        x = r \sin(\theta) \cos(\phi) &\\
        y = r \sin(\theta) \sin(\phi) &\\
        z = r \cos(\theta)
    \end{cases}
    \text{for} \; r \in [0, \infty], \;
    \theta \in [0, \pi], \; \text{and} \;
    \phi \in [0, 2\pi)
    
    
The module also provides functions for handling spherical harmonics.
These spherical harmonics are described by the degree :math:`l` and the order 
:math:`m` or, alternatively, by the mode :math:`k`. The relation between these
values is

.. math::
    k = l(l + 1) + m
    
and

.. math::
    l &= \text{floor}(\sqrt{k}) \\
    m &= k - l(l + 1)
    
We will use these indices interchangeably, although the mode :math:`k` is
preferred internally. Note that we also consider axisymmetric spherical
harmonics, where the order is always zero and the degree :math:`l` and the mode
:math:`k` are thus identical.    
    
    
.. autosummary::
   :nosignatures:

   radius_from_volume
   volume_from_radius
   surface_from_radius
   points_cartesian_to_spherical
   points_spherical_to_cartesian
   haversine_distance
   get_spherical_polygon_area
   PointsOnSphere
   spherical_index_k
   spherical_index_lm
   spherical_index_count
   spherical_index_count_optimal
   spherical_harmonic_symmetric
   spherical_harmonic_real
   spherical_harmonic_real_k
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>    
"""

import itertools
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import spatial
from scipy.special import sph_harm

from ..tools.cache import cached_method
from ..tools.numba import jit

π = np.pi


def radius_from_volume(volume: float, dim: int) -> float:
    """Return the radius of a sphere with a given volume

    Args:
        volume (float): Volume of the sphere
        dim (int): Dimension of the space

    Returns:
        float: Radius of the sphere
    """
    if dim == 1:
        return volume / 2
    elif dim == 2:
        return np.sqrt(volume / π)  # type: ignore
    elif dim == 3:
        return (3 * volume / (4 * π)) ** (1 / 3)  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_radius_from_volume_compiled(dim: int) -> Callable:
    """Return a function calculating the radius of a sphere with a given volume

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a volume and returns the radius
    """
    if dim == 1:

        def radius_from_volume(volume):
            return volume / 2

    elif dim == 2:

        def radius_from_volume(volume):
            return np.sqrt(volume / π)

    elif dim == 3:

        def radius_from_volume(volume):
            return (3 * volume / (4 * π)) ** (1 / 3)

    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")
    return jit(radius_from_volume)  # type: ignore


def volume_from_radius(radius: float, dim: int) -> float:
    """Return the volume of a sphere with a given radius

    Args:
        radius (float): Radius of the sphere
        dim (int): Dimension of the space

    Returns:
        float: Volume of the sphere
    """
    if dim == 1:
        return 2 * radius
    elif dim == 2:
        return π * radius ** 2  # type: ignore
    elif dim == 3:
        return 4 * π / 3 * radius ** 3  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the volume in {dim} dimensions")


def make_volume_from_radius_compiled(dim: int) -> Callable:
    """Return a function calculating the volume of a sphere with a given radius

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a radius and returns the volume
    """
    if dim == 1:

        def volume_from_radius(radius):
            return 2 * radius

    elif dim == 2:

        def volume_from_radius(radius):
            return π * radius ** 2

    elif dim == 3:

        def volume_from_radius(radius):
            return 4 * π / 3 * radius ** 3

    else:
        raise NotImplementedError(f"Cannot calculate the volume in {dim} dimensions")
    return jit(volume_from_radius)  # type: ignore


def surface_from_radius(radius: float, dim: int) -> float:
    """Return the surface area of a sphere with a given radius

    Args:
        radius (float): Radius of the sphere
        dim (int): Dimension of the space

    Returns:
        float: Surface area of the sphere
    """
    if dim == 1:
        return 2
    elif dim == 2:
        return 2 * π * radius  # type: ignore
    elif dim == 3:
        return 4 * π * radius ** 2  # type: ignore
    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )


def radius_from_surface(surface: float, dim: int) -> float:
    """Return the radius of a sphere with a given surface area

    Args:
        surface (float): Surface area of the sphere
        dim (int): Dimension of the space

    Returns:
        float: Radius of the sphere
    """
    if dim == 1:
        raise RuntimeError("Cannot calculate radius of 1-d sphere from surface")
    elif dim == 2:
        return surface / (2 * π)  # type: ignore
    elif dim == 3:
        return np.sqrt(surface / (4 * π))  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_surface_from_radius_compiled(dim: int) -> Callable:
    """Return a function calculating the surface area of a sphere

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a radius and returns the surface area
    """
    if dim == 1:

        def surface_from_radius(radius):
            return 2

    elif dim == 2:

        def surface_from_radius(radius):
            return 2 * π * radius

    elif dim == 3:

        def surface_from_radius(radius):
            return 4 * π * radius ** 2

    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )
    return jit(surface_from_radius)  # type: ignore


def points_cartesian_to_spherical(points):
    """Convert points from Cartesian to spherical coordinates

    Args:
        points (:class:`numpy.ndarray`): Points in Cartesian coordinates

    Returns:
        :class:`numpy.ndarray`: Points (r, θ, φ) in spherical coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3

    ps_spherical = np.empty(points.shape)
    # calculate radius in [0, infinty]
    ps_spherical[..., 0] = np.linalg.norm(points, axis=-1)
    # calculate θ in [0, pi]
    ps_spherical[..., 1] = np.arccos(points[..., 2] / ps_spherical[..., 0])
    # calculate φ in [0, 2 * pi]
    ps_spherical[..., 2] = np.arctan2(points[..., 1], points[..., 0]) % (2 * π)
    return ps_spherical


def points_spherical_to_cartesian(points):
    """Convert points from spherical to Cartesian coordinates

    Args:
        points (:class:`numpy.ndarray`):
            Points in spherical coordinates (r, θ, φ)

    Returns:
        :class:`numpy.ndarray`: Points in Cartesian coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3

    sin_θ = np.sin(points[..., 1])
    ps_cartesian = np.empty(points.shape)
    ps_cartesian[..., 0] = points[..., 0] * np.cos(points[..., 2]) * sin_θ
    ps_cartesian[..., 1] = points[..., 0] * np.sin(points[..., 2]) * sin_θ
    ps_cartesian[..., 2] = points[..., 0] * np.cos(points[..., 1])
    return ps_cartesian


def haversine_distance(point1, point2) -> float:
    """Calculate the haversine-based distance between two points on the surface
    of a sphere. Should be more accurate than the arc cosine strategy.
    See, for example: https://en.wikipedia.org/wiki/Haversine_formula

    Adapted from https://github.com/tylerjereddy/spherical-SA-docker-demo
    Licensed under MIT License (see copy in root of this project)

    Args:
        point1 (:class:`numpy.ndarray`): First point on the sphere (given in
            Cartesian coordinates)
        point2 (:class:`numpy.ndarray`): Second point on the sphere
        radius (float): Radius of the sphere
    """
    # note that latitude φ is θ and longitude λ is φ in our notation
    r1, φ1, λ1 = points_cartesian_to_spherical(point1)
    r2, φ2, λ2 = points_cartesian_to_spherical(point2)

    # check whether both points lie on the same sphere
    assert np.isclose(r1, r2)

    # we rewrite the standard Haversine slightly as long/lat is not the same as
    # spherical coordinates - φ differs by π/4
    factor = (1 - np.cos(λ2 - λ1)) / 2
    arg = (1 - np.cos(φ2 - φ1)) / 2 + np.sin(φ1) * np.sin(φ2) * factor
    return 2 * r1 * np.arcsin(np.sqrt(arg))  # type: ignore


def get_spherical_polygon_area(vertices, radius: float = 1) -> float:
    """Calculate the surface area of a polygon on the surface of a sphere.
    Based on equation provided here:
    http://mathworld.wolfram.com/LHuiliersTheorem.html
    Decompose into triangles, calculate excess for each

    Adapted from https://github.com/tylerjereddy/spherical-SA-docker-demo
    Licensed under MIT License (see copy in root of this project)

    Args:
        vertices (:class:`numpy.ndarray`): List of vertices (using Cartesian
            coordinates) that describe the corners of the polygon. The vertices
            need to be oriented.
        radius (float): Radius of the sphere
    """
    # have to convert to unit sphere before applying the formula
    spherical_coordinates = points_cartesian_to_spherical(vertices)
    spherical_coordinates[..., 0] = 1.0
    vertices = points_spherical_to_cartesian(spherical_coordinates)

    n = vertices.shape[0]
    # point we start from
    root_point = vertices[0]
    totalexcess = 0

    # loop from 1 to n-2, with point 2 to n-1 as other vertex of triangle
    # this could definitely be written more nicely
    b_point = vertices[1]
    root_b_dist = haversine_distance(root_point, b_point)
    for i in np.arange(1, n - 1):
        a_point = b_point
        b_point = vertices[i + 1]
        root_a_dist = root_b_dist
        root_b_dist = haversine_distance(root_point, b_point)
        a_b_dist = haversine_distance(a_point, b_point)
        s = (root_a_dist + root_b_dist + a_b_dist) / 2.0
        arg = (
            np.tan(0.5 * s)
            * np.tan(0.5 * (s - root_a_dist))
            * np.tan(0.5 * (s - root_b_dist))
            * np.tan(0.5 * (s - a_b_dist))
        )
        totalexcess += 4 * np.arctan(np.sqrt(arg))
    return totalexcess * radius ** 2


class PointsOnSphere:
    """ class representing points on an n-dimensional unit sphere """

    def __init__(self, points):
        """
        Args:
            points (:class:`numpy.ndarray`):
                The list of points on the unit sphere
        """
        self.points = np.asarray(points, dtype=np.double)
        # normalize vectors to force them onto the unit-sphere
        self.points /= np.linalg.norm(self.points, axis=1)[:, np.newaxis]
        self.dim = self.points.shape[-1]

    @classmethod
    def make_uniform(cls, dim: int, num_points: Optional[int] = None):
        """create uniformly distributed points on a sphere

        Args:
            dim (int): The dimension of space
            num_points (int, optional): The number of points to generate. Note
                that for one-dimensional spheres (intervals), only exactly two
                points can be generated
        """
        if dim == 1:
            # just have two directions in 2d
            if num_points is None:
                num_points = 2
            if num_points != 2:
                raise ValueError("Can only place 2 points in 1d")
            points = [[-1], [1]]

        elif dim == 2:
            if num_points is None:
                num_points = 8
            # distribute points evenly around the circle
            φs = np.linspace(0, 2 * π, num_points, endpoint=False)
            points = np.c_[np.cos(φs), np.sin(φs)]

        elif dim == 3:
            # Distribute points on the unit sphere using a sunflower spiral
            # (inspired by https://stackoverflow.com/a/44164075/932593)
            if num_points is None:
                num_points = 18
            indices = np.arange(0, num_points) + 0.5
            φ = np.arccos(1 - 2 * indices / num_points)
            θ = π * (1 + 5 ** 0.5) * indices

            # convert to Cartesian coordinates
            points = np.c_[np.cos(θ) * np.sin(φ), np.sin(θ) * np.sin(φ), np.cos(φ)]

        elif num_points is None:
            # use vertices of hypercube in n dimensions
            points = [
                p  # type: ignore
                for p in itertools.product([-1, 0, 1], repeat=dim)
                if any(c != 0 for c in p)
            ]

        else:
            raise NotImplementedError()

        # normalize vectors
        return cls(points)

    @cached_method()
    def get_area_weights(self, balance_axes: bool = True):
        """return the weight of each point associated with the unit cell size

        Args:
            balance_axes (bool): Flag determining whether the weights should be
                chosen such that the weighted average of all points is the
                zero vector

        Returns:
            :class:`numpy.ndarray`: The weight associated with each point
        """
        points_flat = self.points.reshape(-1, self.dim)
        if self.dim == 1:
            weights = np.array([0.5, 0.5])

        elif self.dim == 2:
            # get angles
            φ = np.arctan2(points_flat[:, 1], points_flat[:, 0])
            idx = np.argsort(φ)
            s0 = φ[idx[0]] + 2 * π - φ[idx[-1]]
            sizes = np.r_[s0, np.diff(φ[idx]), s0]
            weights = (sizes[1:] + sizes[:-1]) / 2
            weights /= 2 * π

        elif self.dim == 3:
            # calculate weights using spherical voronoi construction
            voronoi = spatial.SphericalVoronoi(points_flat)
            voronoi.sort_vertices_of_regions()

            weights = [
                get_spherical_polygon_area(voronoi.vertices[ix])
                for ix in voronoi.regions
            ]
            weights = np.array(weights, dtype=np.double)
            weights /= surface_from_radius(1, dim=self.dim)

        else:
            raise NotImplementedError()

        if balance_axes:
            weights /= weights.sum()  # normalize weights
            # adjust weights such that all distances are weighted equally, i.e.,
            # the weighted sum of all shell vectors should vanish. Additionally,
            # the sum of all weights needs to be one. To satisfy these
            # constraints simultaneously, the weights are adjusted minimally
            # (in a least square sense).
            matrix = np.c_[points_flat, np.ones(len(points_flat))]
            vector = -weights @ matrix + np.r_[np.zeros(self.dim), 1]
            weights += np.linalg.lstsq(matrix.T, vector, rcond=None)[0]

        return weights.reshape(self.points.shape[:-1])

    def get_distance_matrix(self):
        """calculate the (spherical) distances between each point

        Returns:
            :class:`numpy.ndarray`: the distance of each point to each other
        """
        if self.dim == 1:
            raise ValueError("Distances can only be calculated for dim >= 2")

        elif self.dim == 2:
            # use arc length on unit circle to calculate distances
            def metric(a, b):
                return np.arccos(a @ b)

        elif self.dim == 3:
            # calculate distances on sphere using haversine definition
            metric = haversine_distance

        else:
            raise NotImplementedError()

        # determine the distances between all points
        dists = spatial.distance.pdist(self.points, metric)
        return spatial.distance.squareform(dists)

    def get_mean_separation(self) -> float:
        """ float: calculates the mean distance to the nearest neighbor """
        if len(self.points) < 1:
            return float("nan")

        dists_sorted = np.sort(self.get_distance_matrix(), axis=1)
        return float(dists_sorted[:, 1].mean())

    def write_to_xyz(self, path: str, comment: str = "", symbol: str = "S"):
        """write the point coordinates to a xyz file

        Args:
            filename (str): location of the file where data is written
            comment (str, optional): comment that is written to the second line
            symbol (str, optional): denotes the symbol used for the atoms
        """
        with open(path, "w") as fp:
            fp.write("%d\n" % len(self.points))
            fp.write(comment + "\n")
            for point in self.points:
                point_str = " ".join(("%.12g" % v for v in point))
                line = "%s %s\n" % (symbol, point_str)
                fp.write(line)


def spherical_index_k(degree: int, order: int = 0) -> int:
    """returns the mode `k` from the degree `degree` and order `order`

    Args:
        degree (int): Degree of the spherical harmonics
        order (int): Order of the spherical harmonics

    Raises:
        ValueError: if `order < -degree` or `order > degree`

    Returns:
        int: a combined index k
    """
    if not -degree <= order <= degree:
        raise ValueError("order must lie between -degree and degree")
    return degree * (degree + 1) + order


def spherical_index_lm(k: int) -> Tuple[int, int]:
    """returns the degree `l` and the order `m` from the mode `k`

    Args:
        k (int): The combined index for the spherical harmonics

    Returns:
        tuple: The degree `l` and order `m` of the spherical harmonics
        assoicated with the combined index
    """
    degree = int(np.floor(np.sqrt(k)))
    return degree, k - degree * (degree + 1)


def spherical_index_count(l: int) -> int:
    """return the number of modes for all indices <= l

    The returned value is one less than the maximal mode `k` required.

    Args:
        degree (int): Maximal degree of the spherical harmonics

    Returns:
        int: The number of modes
    """
    return 1 + 2 * l + l * l


def spherical_index_count_optimal(k_count: int) -> bool:
    """checks whether the modes captures all orders for maximal degree

    Args:
        k_count (int): The number of modes considered
    """
    is_square = bool(int(np.sqrt(k_count) + 0.5) ** 2 == k_count)
    return is_square


def spherical_harmonic_symmetric(degree: int, θ: float) -> float:
    r"""axisymmetric spherical harmonics with degree `degree`, so `m=0`.

    Args:
        degree (int): Degree of the spherical harmonics
        θ (float): Azimuthal angle at which the spherical harmonics is
            evaluated (in :math:`[0, \pi]`)

    Returns:
        float: The value of the spherical harmonics
    """
    # note that the definition of `sph_harm` has a different convention for the
    # usage of the variables φ and θ and we thus have to swap the args
    return np.real(sph_harm(0.0, degree, 0.0, θ))  # type: ignore


def spherical_harmonic_real(degree: int, order: int, θ: float, φ: float) -> float:
    r"""real spherical harmonics of degree l and order m

    Args:
        degree (int): Degree :math:`l` of the spherical harmonics
        order (int): Order :math:`m` of the spherical harmonics
        θ (float): Azimuthal angle (in :math:`[0, \pi]`) at which the
            spherical harmonics is evaluated.
        φ (float): Polar angle (in :math:`[0, 2\pi]`) at which the spherical
            harmonics is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    # note that the definition of `sph_harm` has a different convention for the
    # usage of the variables φ and θ and we thus have to swap the args
    # Moreover, the scipy functions expect first the order and then the degree
    if order > 0:
        term1 = sph_harm(order, degree, φ, θ)
        term2 = (-1) ** order * sph_harm(-order, degree, φ, θ)
        return np.real((term1 + term2) / np.sqrt(2))  # type: ignore

    elif order == 0:
        return np.real(sph_harm(0, degree, φ, θ))  # type: ignore

    else:  # order < 0
        term1 = sph_harm(-order, degree, φ, θ)
        term2 = (-1) ** order * sph_harm(order, degree, φ, θ)
        return np.real((term1 - term2) / (np.complex(0, np.sqrt(2))))  # type: ignore


def spherical_harmonic_real_k(k: int, θ: float, φ: float) -> float:
    r"""real spherical harmonics described by mode k

    Args:
        k (int): Combined index determining the degree and order of the
            spherical harmonics
        θ (float): Azimuthal angle (in :math:`[0, \pi]`) at which the
            spherical harmonics is evaluated.
        φ (float): Polar angle (in :math:`[0, 2\pi]`) at which the spherical
            harmonics is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    return spherical_harmonic_real(*spherical_index_lm(k), θ=θ, φ=φ)
