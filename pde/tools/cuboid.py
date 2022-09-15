"""
An n-dimensional, axes-aligned cuboid

This module defines the :class:`Cuboid` class, which represents an n-dimensional
cuboid that is aligned with the axes of a Cartesian coordinate system.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools
from typing import List, Tuple

import numpy as np

from .typing import FloatNumerical


class Cuboid:
    """class that represents a cuboid in :math:`n` dimensions"""

    def __init__(self, pos, size, mutable: bool = True):
        """defines a cuboid from a position and a size vector

        Args:
            pos (list):
                The position of the lower left corner. The length of this list
                determines the dimensionality of space
            size (list):
                The size of the cuboid along each dimension.
            mutable (bool):
                Flag determining whether the cuboid parameters can be changed
        """
        self._mutable = mutable
        # set position and adjust mutable status later
        self.pos = np.array(pos, copy=True)
        self.size = size  # implicitly sets correct shape
        self.pos.flags.writeable = self.mutable

    @property
    def size(self) -> np.ndarray:
        return self._size

    @size.setter
    def size(self, value: FloatNumerical):
        self._size = np.array(value, self.pos.dtype)  # make copy
        if self.pos.shape != self._size.shape:
            raise ValueError(
                f"Size vector (dim={len(self._size)}) must have the same "
                f"dimension as the position vector (dim={len(self.pos)})"
            )

        # flip Cuboid with negative size
        neg = self._size < 0
        self.pos[neg] += self._size[neg]
        self._size = np.abs(self._size)
        self._size.flags.writeable = self.mutable

    @property
    def corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """return coordinates of two extreme corners defining the cuboid"""
        return np.copy(self.pos), self.pos + self.size

    @property
    def mutable(self) -> bool:
        return self._mutable

    @mutable.setter
    def mutable(self, value: bool):
        self._mutable = bool(value)
        self.pos.flags.writeable = self._mutable
        self._size.flags.writeable = self._mutable

    @classmethod
    def from_points(cls, p1: np.ndarray, p2: np.ndarray, **kwargs) -> "Cuboid":
        """create cuboid from two points

        Args:
            p1 (list): Coordinates of first corner point
            p2 (list): Coordinates of second corner point

        Returns:
            Cuboid: cuboid with positive size
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return cls(p1, p2 - p1, **kwargs)

    @classmethod
    def from_bounds(cls, bounds: np.ndarray, **kwargs) -> "Cuboid":
        """create cuboid from bounds

        Args:
            bounds (list): Two dimensional array of axes bounds

        Returns:
            Cuboid: cuboid with positive size
        """
        bounds = np.asarray(bounds).reshape(-1, 2)
        return cls(bounds[:, 0], bounds[:, 1] - bounds[:, 0], **kwargs)

    @classmethod
    def from_centerpoint(
        cls, centerpoint: np.ndarray, size: np.ndarray, **kwargs
    ) -> "Cuboid":
        """create cuboid from two points

        Args:
            centerpoint (list): Coordinates of the center
            size (list): Size of the cuboid

        Returns:
            Cuboid: cuboid with positive size
        """
        centerpoint = np.asarray(centerpoint)
        size = np.asarray(size)
        return cls(centerpoint - size / 2, size, **kwargs)

    def copy(self) -> "Cuboid":
        return self.__class__(self.pos, self.size)

    def __repr__(self):
        return "{cls}(pos={pos}, size={size})".format(
            cls=self.__class__.__name__, pos=self.pos, size=self.size
        )

    def __add__(self, other: "Cuboid") -> "Cuboid":
        """The sum of two cuboids is the minimal cuboid enclosing both"""
        if isinstance(other, Cuboid):
            if self.dim != other.dim:
                raise RuntimeError("Incompatible dimensions")
            a1, a2 = self.corners
            b1, b2 = other.corners
            return self.__class__.from_points(np.minimum(a1, b1), np.maximum(a2, b2))

        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        """override the default equality test"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return np.all(self.pos == other.pos) and np.all(self.size == other.size)  # type: ignore

    @property
    def dim(self) -> int:
        return len(self.pos)

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((p, p + s) for p, s in zip(self.pos, self.size))

    @property
    def vertices(self) -> List[List[float]]:
        """return the coordinates of all the corners"""
        return list(itertools.product(*self.bounds))  # type: ignore

    @property
    def diagonal(self) -> float:
        """returns the length of the diagonal"""
        return np.linalg.norm(self.size)  # type: ignore

    @property
    def surface_area(self) -> float:
        """surface area of a cuboid in :math:`n` dimensions.

        The surface area is the volume of the (:math:`n-1`)-dimensional
        hypercubes that bound the current cuboid:

            * :math:`n=1`: the number of end points (2)
            * :math:`n=2`: the perimeter of the rectangle
            * :math:`n=3`: the surface area of the cuboid
        """
        sides = self.size
        null = sides == 0
        null_count = null.sum()
        if null_count == 0:
            return 2 * np.sum(np.prod(sides) / sides)  # type: ignore
        elif null_count == 1:
            return 2 * np.prod(sides[~null])  # type: ignore
        else:
            return 0

    @property
    def centroid(self):
        return self.pos + self.size / 2

    @centroid.setter
    def centroid(self, center):
        self.pos[:] = np.asanyarray(center) - self.size / 2

    @property
    def volume(self) -> float:
        return np.prod(self.size)  # type: ignore

    def buffer(self, amount: FloatNumerical = 0, inplace=False) -> "Cuboid":
        """dilate the cuboid by a certain amount in all directions"""
        amount = np.asarray(amount)
        if inplace:
            self.pos -= amount
            self.size += 2 * amount
            return self
        else:
            return self.__class__(self.pos - amount, self.size + 2 * amount)

    def contains_point(self, points: np.ndarray) -> np.ndarray:
        """returns a True when `points` are within the Cuboid

        Args:
            points (:class:`~numpy.ndarray`): List of point coordinates

        Returns:
            :class:`~numpy.ndarray`: list of booleans indicating which points are inside
        """
        points = np.asarray(points)
        if len(points) == 0:
            return points

        if points.shape[-1] != self.dim:
            raise ValueError(
                "Last dimension of `points` must agree with "
                f"cuboid dimension {self.dim}"
            )

        c1, c2 = self.corners
        return np.all(c1 <= points, axis=-1) & np.all(points <= c2, axis=-1)  # type: ignore


def asanyarray_flags(data: np.ndarray, dtype=None, writeable: bool = True):
    """turns data into an array and sets the respective flags.

    A copy is only made if necessary

    Args:
        data (:class:`~numpy.ndarray`): numpy array that whose flags are adjusted
        dtype: the resulant dtype
        writeable (bool): Flag determining whether the results is writable

    Returns:
        :class:`~numpy.ndarray`:
            array with same data as `data` but with flags adjusted.
    """
    try:
        data_writeable = data.flags.writeable
    except AttributeError:
        # `data` did not have the writeable flag => it's not a numpy array
        result = np.array(data, dtype)
    else:
        if data_writeable != writeable:
            # need to make a copy because the flags differ
            result = np.array(data, dtype)
        else:
            # might have to make a copy to adjust the dtype
            result = np.asanyarray(data, dtype)

    # set the flags and return the array
    result.flags.writeable = writeable
    return result
