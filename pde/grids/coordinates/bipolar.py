"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import CoordinatesBase


class BipolarCoordinates(CoordinatesBase):
    """2-dimensional bipolar coordinates"""

    dim = 2
    axes = ["σ", "τ"]
    _axes_alt = {"σ": ["sigma"], "τ": ["tau"]}
    coordinate_limits = [(0, 2 * np.pi), (-np.inf, np.inf)]

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
        σ, τ = points[..., 0], points[..., 1]
        denom = np.cosh(τ) - np.cos(σ)
        x = self.scale_parameter * np.sinh(τ) / denom
        y = self.scale_parameter * np.sin(σ) / denom
        return np.stack((x, y), axis=-1)

    def _pos_from_cart(self, points: np.ndarray) -> np.ndarray:
        x, y = points[..., 0], points[..., 1]
        a = self.scale_parameter
        h2 = x**2 + y**2
        denom = a**2 - h2 + np.sqrt((a**2 - h2) ** 2 + 4 * a**2 * y**2)
        σ = np.mod(np.pi - 2 * np.arctan2(2 * a * y, denom), 2 * np.pi)
        τ = 0.5 * np.log(((x + a) ** 2 + y**2) / ((x - a) ** 2 + y**2))
        return np.stack((σ, τ), axis=-1)

    def _mapping_jacobian(self, points: np.ndarray) -> np.ndarray:
        σ, τ = points[..., 0], points[..., 1]

        sinσ = np.sin(σ)
        cosσ = np.cos(σ)
        sinhτ = np.sinh(τ)
        coshτ = np.cosh(τ)
        factor = self.scale_parameter * (cosσ - coshτ) ** -2

        return factor * np.array(  # type: ignore
            [
                [-sinσ * sinhτ, 1 - cosσ * coshτ],
                [cosσ * coshτ - 1, -sinσ * sinhτ],
            ]
        )

    def _volume_factor(self, points: np.ndarray) -> ArrayLike:
        σ, τ = points[..., 0], points[..., 1]
        return self.scale_parameter**2 * (np.cosh(τ) - np.cos(σ)) ** -2  # type: ignore

    def _scale_factors(self, points: np.ndarray) -> np.ndarray:
        σ, τ = points[..., 0], points[..., 1]
        sf = self.scale_parameter / (np.cosh(τ) - np.cos(σ))
        return np.array([sf, sf])

    def _basis_rotation(self, points: np.ndarray) -> np.ndarray:
        σ, τ = points[..., 0], points[..., 1]

        sinσ = np.sin(σ)
        cosσ = np.cos(σ)
        sinhτ = np.sinh(τ)
        coshτ = np.cosh(τ)
        factor = 1 / (cosσ - coshτ)

        return factor * np.array(  # type: ignore
            [
                [sinσ * sinhτ, 1 - cosσ * coshτ],
                [cosσ * coshτ - 1, sinσ * sinhτ],
            ]
        )
