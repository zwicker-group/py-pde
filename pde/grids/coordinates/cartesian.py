"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class CartesianCoordinates(CoordinatesBase):
    """n-dimensional Cartesian coordinates"""

    _objs: dict[int, CartesianCoordinates] = {}

    def __new__(cls, dim: int):
        # cache the instances for each dimension
        if dim not in cls._objs:
            cls._objs[dim] = super().__new__(cls)
        return cls._objs[dim]

    def __getnewargs__(self):
        return (self.dim,)

    def __init__(self, dim: int):
        """
        Args:
            dim (int):
                Dimension of the Cartesian coordinate system
        """
        if dim <= 0:
            raise ValueError("`dim` must be positive integer")
        self.dim = dim
        if self.dim <= 3:
            self.axes = list("xyz"[: self.dim])
        else:
            self.axes = [chr(97 + i) for i in range(self.dim)]
        self.coordinate_limits = [(-np.inf, np.inf)] * self.dim

    def __repr__(self) -> str:
        """return instance as string"""
        return f"{self.__class__.__name__}(dim={self.dim})"

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.dim == other.dim

    def _pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        return points

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        return points

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        jac = np.zeros((self.dim, self.dim) + points.shape[:-1])
        jac[range(self.dim), range(self.dim)] = 1
        return jac

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        return np.ones(points.shape[:-1])

    def _cell_volume(self, c_low: np.ndarray, c_high: np.ndarray):
        return np.prod(c_high - c_low, axis=-1)

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        return np.ones_like(points)

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        return np.eye(self.dim)
