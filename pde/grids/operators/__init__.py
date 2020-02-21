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

# Package-wide constant defining when to use parallel numba 
PARALLELIZATION_THRESHOLD_2D = 256
""" int: threshold for determining when parallel code is created for
differential operators. The value gives the minimal number of support points in
each direction for a 2-dimensional grid """
PARALLELIZATION_THRESHOLD_3D = 64
""" int: threshold for determining when parallel code is created for
differential operators. The value gives the minimal number of support points in
each direction for a 3-dimensional grid """
