"""Package collecting modules defining discretized operators using torch.

These operators can either be used directly or they are imported by the  respective
methods defined on fields and grids.

.. autosummary::
   :nosignatures:

   cartesian
   polar_sym
"""

from . import cartesian, polar_sym, spherical_sym  # noqa: F401
