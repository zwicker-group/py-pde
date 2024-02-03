"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class SphericalCoordinates(CoordinatesBase):
    """3-dimensional spherical coordinates"""

    dim = 3
    axes = ["r", "θ", "φ"]
    _axes_alt = {"θ": ["theta"], "φ": ["phi"]}
    coordinate_limits = [(0, np.inf), (0, np.pi), (0, 2 * np.pi)]
    major_axis = 0

    _singleton: SphericalCoordinates | None = None

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
        r, θ, φ = points[..., 0], points[..., 1], points[..., 2]
        rsinθ = r * np.sin(θ)
        x = rsinθ * np.cos(φ)
        y = rsinθ * np.sin(φ)
        z = r * np.cos(θ)
        return np.stack((x, y, z), axis=-1)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = np.linalg.norm(points, axis=-1)
        θ = np.arctan2(np.hypot(x, y), z)
        φ = np.arctan2(y, x)
        return np.stack((r, θ, φ), axis=-1)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        r, θ, φ = points[..., 0], points[..., 1], points[..., 2]
        sinθ, cosθ = np.sin(θ), np.cos(θ)
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        return np.array(
            [
                [cosφ * sinθ, r * cosφ * cosθ, -r * sinφ * sinθ],
                [sinφ * sinθ, r * sinφ * cosθ, r * cosφ * sinθ],
                [cosθ, -r * sinθ, np.zeros_like(θ)],
            ]
        )

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        r, θ = points[..., 0], points[..., 1]
        return r**2 * np.sin(θ)  # type: ignore

    def _cell_volume(self, c_low: np.ndarray, c_high: np.ndarray):
        r1, θ1, φ1 = c_low[..., 0], c_low[..., 1], c_low[..., 2]
        r2, θ2, φ2 = c_high[..., 0], c_high[..., 1], c_high[..., 2]
        return (φ2 - φ1) * (np.cos(θ1) - np.cos(θ2)) * (r2**3 - r1**3) / 3

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        r, θ = points[..., 0], points[..., 1]
        return np.array([np.ones_like(r), r, r * np.sin(θ)])

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        θ, φ = points[..., 1], points[..., 2]
        sinθ, cosθ = np.sin(θ), np.cos(θ)
        sinφ, cosφ = np.sin(φ), np.cos(φ)
        return np.array(
            [
                [cosφ * sinθ, sinφ * sinθ, cosθ],
                [cosφ * cosθ, sinφ * cosθ, -sinθ],
                [-sinφ, cosφ, np.zeros_like(θ)],
            ]
        )
