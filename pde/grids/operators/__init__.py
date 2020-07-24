"""
Package collecting modules defining discretized operators for different grids.

These operators can either be used directly or they are imported by the 
respective methods defined on fields and grids.


.. autosummary::
   :nosignatures:

   cartesian
   cylindrical
   polar
   spherical
"""

from . import cartesian, cylindrical, polar, spherical
