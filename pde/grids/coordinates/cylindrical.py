"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class CylindricalCoordinates(CoordinatesBase):
    """n-dimensional Cartesian coordinates"""

    _singleton: CylindricalCoordinates | None = None
    dim = 3
    coordinate_limits = [(0, np.inf), (0, 2 * np.pi), (-np.inf, np.inf)]
    axes = ["r", "φ", "z"]
    _axes_alt = {"φ": ["phi"]}

    def __new__(cls):
        # cache the instances for each dimension
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def _pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        r, φ, z = points[..., 0], points[..., 1], points[..., 2]
        x = r * np.cos(φ)
        y = r * np.sin(φ)
        return np.stack((x, y, z), axis=-1)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = np.hypot(x, y)
        φ = np.arctan2(y, x)
        return np.stack((r, φ, z), axis=-1)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        r, φ = points[..., 0], points[..., 1]
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        zero = np.zeros_like(r)
        return np.array(
            [
                [cosφ, -r * sinφ, zero],
                [sinφ, r * cosφ, zero],
                [zero, zero, zero + 1],
            ]
        )

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        return points[..., 0]

    def _cell_volume(self, c_low: np.ndarray, c_high: np.ndarray):
        r1, φ1, z1 = c_low[..., 0], c_low[..., 1], c_low[..., 2]
        r2, φ2, z2 = c_high[..., 0], c_high[..., 1], c_high[..., 2]
        return (φ2 - φ1) * (z2 - z1) * (r2**2 - r1**2) / 2

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        r = points[..., 0]
        ones = np.ones_like(r)
        return np.array([ones, r, ones])

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        φ = points[..., 1]
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        zero = np.zeros_like(φ)
        return np.array(
            [
                [cosφ, sinφ, zero],
                [-sinφ, cosφ, zero],
                [zero, zero, zero + 1],
            ]
        )
