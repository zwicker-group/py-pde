"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, optimize


class DimensionError(ValueError):
    """exception indicating that dimensions were inconsistent"""


class CoordinatesBase:
    """Base class for orthonormal coordinate systems"""

    # properties that are defined in subclasses
    dim: int
    """int: spatial dimension of the coordinate system"""
    coordinate_limits: list[tuple[float, float]]
    """list of tuple: the limits of each coordinate axis"""
    axes: list[str]
    """list: name of each coordinate axis"""
    _axes_alt: dict[str, list[str]] = {}
    """dict: maps alternative names for axes to the canonical ones"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def _axes_alt_repl(self) -> dict[str, str]:
        """dict: replacement rules for axes names"""
        res = {}
        for axes_name, repl_list in self._axes_alt.items():
            for axes_alt in repl_list:
                res[axes_alt] = axes_name
        return res

    def _pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        # actual calculation needs to be implemented by sub-class
        raise NotImplementedError

    def pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        """convert coordinates to Cartesian coordinates

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of points in the current coordinate system

        Returns:
            :class:`~numpy.ndarray`: Cartesian coordinates of the points
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._pos_to_cart(points)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        # actual calculation needs to be implemented by sub-class
        raise NotImplementedError

    def pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        """convert Cartesian coordinates to coordinates in this system

        Args:
            points (:class:`~numpy.ndarray`):
                Points given in Cartesian coordinates.

        Returns:
            :class:`~numpy.ndarray`: Points given in the coordinates of this system
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._pos_from_cart(points)

    def pos_diff(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """return Cartesian vector(s) pointing from p1 to p2

        Args:
            p1 (:class:`~numpy.ndarray`):
                First point(s)
            p2 (:class:`~numpy.ndarray`):
                Second point(s)

        Returns:
            :class:`~numpy.ndarray`: The difference vectors between the points with
            periodic boundary conditions applied.
        """
        # deprecated since 2024-01-28
        warnings.warn(
            "Deprecated function. Use `pos_to_cart` instead", DeprecationWarning
        )
        return self.pos_to_cart(p2) - self.pos_to_cart(p1)  # type: ignore

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate the distance between two points

        Args:
            p1 (:class:`~numpy.ndarray`):
                First position
            p2 (:class:`~numpy.ndarray`):
                Second position

        Returns:
            float: Distance between the two positions
        """
        # this can be overwritten if a more efficient calculation is possible
        x1 = self.pos_to_cart(p1)
        x2 = self.pos_to_cart(p2)
        return np.linalg.norm(x2 - x1, axis=-1)  # type: ignore

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        return np.diag(self.metric(points)) ** 2

    def scale_factors(self, points: np.ndarray) -> np.ndarray:
        """calculate the scale factors at various points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the points

        Returns:
            :class:`~numpy.ndarray`: Scale factors at the points
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._scale_factors(points)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        # Basic implementation based on finite difference, which should be overwritten
        # using analytical expressions for speed and accuracy
        jac = np.apply_along_axis(
            lambda p: optimize.approx_fprime(p, self.pos_to_cart), 0, points
        )
        if self.dim == 1 and jac.ndim != points.ndim + 1:
            # this happens with some versions of scipy, which collapses dimensions
            jac = jac[..., np.newaxis]
        return jac

    def mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        """returns the Jacobian matrix of the coordinate mapping

        Args:
            points (:class:`~numpy.ndarray`):
                Coordinates of the point(s)

        Returns:
            :class:`~numpy.ndarray`: The Jacobian
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._mapping_jacobian(points)

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        # default implementation based on scale factors
        return np.prod(self._scale_factors(points), axis=0)  # type: ignore

    def volume_factor(self, points: np.ndarray) -> ArrayLike:
        """calculate the volume factors at various points

        Args:
            points (:class:`~numpy.ndarray`):
                Coordinates of the point(s)

        Returns:
            :class:`~numpy.ndarray`: Volume factors at the points
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._volume_factor(points)

    def _cell_volume(self, c_low: np.ndarray, c_high: np.ndarray) -> np.ndarray:
        # Basic implementation based on numerical integration, which should be
        # overwritten if the integration can be done analytically
        cell_volumes = np.empty(c_low.shape[:-1])
        for i in np.ndindex(*cell_volumes.shape):
            cell_volumes[i] = integrate.nquad(
                lambda *x: self._volume_factor(np.array(x)), np.c_[c_low[i], c_high[i]]
            )[0]
        return cell_volumes

    def cell_volume(self, c_low: np.ndarray, c_high: np.ndarray) -> np.ndarray:
        """calculate the volume between coordinate lines

        Args:
            c_low (:class:`~numpy.ndarray`):
                Lower values of the coordinate lines enclosing the volume
            c_high (:class:`~numpy.ndarray`):
                Upper values of the coordinate lines enclosing the volume

        Returns:
            :class:`~numpy.ndarray`: Enclosed volumes for all given cells
        """
        c_low = np.atleast_1d(c_low)
        if c_low.shape[-1] != self.dim:
            raise DimensionError(f"Shape {c_low.shape} cannot denote points")
        c_high = np.atleast_1d(c_high)
        if c_high.shape[-1] != self.dim:
            raise DimensionError(f"Shape {c_high.shape} cannot denote points")
        return self._cell_volume(c_low, c_high)

    def metric(self, points: np.ndarray) -> np.ndarray:
        """calculate the metric tensor at coordinate points

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the points

        Returns:
            :class:`~numpy.ndarray`: Metric tensor at the points
        """
        # This general implementation assumes that the metric is diagnoal!
        points = np.atleast_1d(points)
        metric = np.zeros((self.dim, self.dim) + points.shape[:-1])
        metric[range(self.dim), range(self.dim)] = self.scale_factors(points) ** 2
        return metric

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def basis_rotation(self, points: np.ndarray) -> np.ndarray:
        """returns rotation matrix rotating basis vectors to Cartesian coordinates

        Args:
            points (:class:`~numpy.ndarray`):
                Coordinates of the point(s)

        Returns:
            :class:`~numpy.ndarray`: Rotation matrices for all points. The returnd array
            has the shape `(dim, dim) + points_shape`, assuming `points` has the shape
            `points_shape + (dim,)`.
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        return self._basis_rotation(points)

    def vec_to_cart(self, points: np.ndarray, components: np.ndarray) -> np.ndarray:
        """convert the vectors at given points to a Cartesian basis

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the point(s) where the vectors are specified.
            components (:class:`~numpy.ndarray`):
                The components of the vectors at the given points

        Returns:
            :class:`~numpy.ndarray`: The vectors specified at the same position but with
            components given in Cartesian coordinates.
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Shape {points.shape} cannot denote points")
        shape = points.shape[:-1]  # shape of array describing the different points
        vec_shape = (self.dim,) + shape

        components = np.atleast_1d(components)
        if components.shape != vec_shape:
            raise DimensionError(f"`components` must have shape {vec_shape}")

        # convert the basis of the vectors to Cartesian
        basis = self.basis_rotation(points)
        assert basis.shape == (self.dim, self.dim) + shape
        return np.einsum("j...,ji...->i...", components, basis)  # type: ignore
