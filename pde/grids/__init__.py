"""Grids define the domains on which PDEs will be solved. In particular, symmetries,
periodicities, and the discretizations are defined by the underlying grid.

We only consider regular, orthogonal grids, which are constructed from orthogonal
coordinate systems with equidistant discretizations along each axis. The dimension of
the space that the grid describes is given by the attribute :attr:`dim`. Cartesian
coordinates can be mapped to grid coordinates and the corresponding discretization cells
using the method :meth:`transform`.

.. autosummary::
   :nosignatures:

   ~cartesian.UnitGrid
   ~cartesian.CartesianGrid
   ~spherical.PolarSymGrid
   ~spherical.SphericalSymGrid
   ~cylindrical.CylindricalSymGrid

Inheritance structure of the classes:

.. inheritance-diagram::
      cartesian.UnitGrid
      cartesian.CartesianGrid
      spherical.PolarSymGrid
      spherical.SphericalSymGrid
      cylindrical.CylindricalSymGrid
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import DimensionError, GridBase  # noqa: F401
from .boundaries import *  # noqa: F403
from .cartesian import CartesianGrid, UnitGrid  # noqa: F401
from .cylindrical import CylindricalSymGrid  # noqa: F401
from .spherical import PolarSymGrid, SphericalSymGrid  # noqa: F401
