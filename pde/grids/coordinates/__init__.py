"""Package collecting classes representing orthonormal coordinate systems.

.. autosummary::
   :nosignatures:

    ~bipolar.BipolarCoordinates
    ~bispherical.BisphericalCoordinates
    ~cartesian.CartesianCoordinates
    ~cylindrical.CylindricalCoordinates
    ~polar.PolarCoordinates
    ~spherical.SphericalCoordinates
"""

from .base import CoordinatesBase, DimensionError  # noqa: F401
from .bipolar import BipolarCoordinates
from .bispherical import BisphericalCoordinates
from .cartesian import CartesianCoordinates
from .cylindrical import CylindricalCoordinates
from .polar import PolarCoordinates
from .spherical import SphericalCoordinates

__all__ = [
    "BipolarCoordinates",
    "BisphericalCoordinates",
    "CartesianCoordinates",
    "CylindricalCoordinates",
    "PolarCoordinates",
    "SphericalCoordinates",
]
