"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class PolarCoordinates(CoordinatesBase):
    """2-dimensional polar coordinates"""

    dim = 2
    axes = ["r", "φ"]
    _axes_alt = {"φ": ["phi"]}
    coordinate_limits = [(0, np.inf), (0, 2 * np.pi)]

    _singleton: PolarCoordinates | None = None

    def __new__(cls):
        # cache the instances for each dimension
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:
        """return instance as string"""
        return f"{self.__class__.__name__}()"

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def _pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        r, φ = points[..., 0], points[..., 1]
        x = r * np.cos(φ)
        y = r * np.sin(φ)
        return np.stack((x, y), axis=-1)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        x, y = points[..., 0], points[..., 1]
        r = np.hypot(x, y)
        φ = np.arctan2(y, x)
        return np.stack((r, φ), axis=-1)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        r, φ = points[..., 0], points[..., 1]
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        return np.array([[cosφ, -r * sinφ], [sinφ, r * cosφ]])

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        return points[..., 0]

    def _cell_volume(self, c_low: np.ndarray, c_high: np.ndarray) -> np.ndarray:
        r1, φ1 = c_low[..., 0], c_low[..., 1]
        r2, φ2 = c_high[..., 0], c_high[..., 1]
        return (φ2 - φ1) * (r2**2 - r1**2) / 2  # type: ignore

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        r = points[..., 0]
        return np.array([np.ones_like(r), r])

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        φ = points[..., 1]
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        return np.array([[cosφ, sinφ], [-sinφ, cosφ]])
