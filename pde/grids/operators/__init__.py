"""
Package collecting modules defining discretized operators for different grids.

These operators can either be used directly or they are imported by the  respective
methods defined on fields and grids.

.. autosummary::
   :nosignatures:

   cartesian
   cylindrical_sym
   polar_sym
   spherical_sym
"""

from . import cartesian, cylindrical_sym, polar_sym, spherical_sym
