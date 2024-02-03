"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class BisphericalCoordinates(CoordinatesBase):
    """3-dimensional bispherical coordinates"""

    dim = 3
    axes = ["σ", "τ", "φ"]
    _axes_alt = {"σ": ["sigma"], "τ": ["tau"], "φ": ["phi"]}
    coordinate_limits = [(0, np.pi), (-np.inf, np.inf), (0, 2 * np.pi)]

    def __init__(self, scale_parameter: float = 1):
        super().__init__()
        if scale_parameter <= 0:
            raise ValueError("Scale parameter must be positive")
        self.scale_parameter = scale_parameter

    def __repr__(self) -> str:
        """return instance as string"""
        return f"{self.__class__.__name__}(scale_parameter={self.scale_parameter})"

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self.scale_parameter == other.scale_parameter
        )

    def _pos_to_cart(self, points: np.ndarray) -> np.ndarray:
        σ, τ, φ = points[..., 0], points[..., 1], points[..., 2]
        denom = np.cosh(τ) - np.cos(σ)
        x = self.scale_parameter * np.sin(σ) / denom * np.cos(φ)
        y = self.scale_parameter * np.sin(σ) / denom * np.sin(φ)
        z = self.scale_parameter * np.sinh(τ) / denom
        return np.stack((x, y, z), axis=-1)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        d = np.hypot(x, y)  # denotes the distance from the z-axis
        a = self.scale_parameter
        h2 = z**2 + d**2
        denom = a**2 - h2 + np.sqrt((a**2 - h2) ** 2 + 4 * a**2 * d**2)
        σ = np.pi - 2 * np.arctan2(2 * a * d, denom)
        τ = 0.5 * np.log(((z + a) ** 2 + d**2) / ((z - a) ** 2 + d**2))
        φ = np.arctan2(y, x)
        return np.stack((σ, τ, φ), axis=-1)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        σ, τ, φ = points[..., 0], points[..., 1], points[..., 2]

        sinσ = np.sin(σ)
        cosσ = np.cos(σ)
        sinhτ = np.sinh(τ)
        coshτ = np.cosh(τ)
        sinφ = np.sin(φ)
        cosφ = np.cos(φ)
        d = cosσ - coshτ
        factor = self.scale_parameter * d**-2

        return factor * np.array(  # type: ignore
            [
                [cosφ * (cosσ * coshτ - 1), -cosφ * sinσ * sinhτ, sinφ * sinσ * d],
                [sinφ * (cosσ * coshτ - 1), -sinφ * sinσ * sinhτ, -cosφ * sinσ * d],
                [-sinσ * sinhτ, 1 - cosσ * coshτ, np.zeros_like(σ)],
            ]
        )

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        σ, τ = points[..., 0], points[..., 1]
        return self.scale_parameter**3 * np.sin(σ) * (np.cosh(τ) - np.cos(σ)) ** -3  # type: ignore

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        σ, τ = points[..., 0], points[..., 1]
        sf = self.scale_parameter / (np.cosh(τ) - np.cos(σ))
        return np.array([sf, sf, sf * np.sin(σ)])

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        σ, τ, φ = points[..., 0], points[..., 1], points[..., 2]

        sinσ = np.sin(σ)
        cosσ = np.cos(σ)
        sinhτ = np.sinh(τ)
        coshτ = np.cosh(τ)
        sinφ = np.sin(φ)
        cosφ = np.cos(φ)
        d = cosσ - coshτ

        return np.array(
            [
                [
                    cosφ * (1 - cosσ * coshτ) / d,
                    sinφ * (1 - cosσ * coshτ) / d,
                    sinσ * sinhτ / d,
                ],
                [
                    cosφ * sinσ * sinhτ / d,
                    sinφ * sinσ * sinhτ / d,
                    (cosσ * coshτ - 1) / d,
                ],
                [-sinφ, cosφ, np.zeros_like(σ)],
            ]
        )
