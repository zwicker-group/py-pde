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
from .bipolar import BipolarCoordinates  # noqa: F401
from .bispherical import BisphericalCoordinates  # noqa: F401
from .cartesian import CartesianCoordinates  # noqa: F401
from .cylindrical import CylindricalCoordinates  # noqa: F401
from .polar import PolarCoordinates  # noqa: F401
from .spherical import SphericalCoordinates  # noqa: F401
