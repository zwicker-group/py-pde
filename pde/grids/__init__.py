"""
Grids define the domains on which PDEs will be solved. In particular, symmetries,
periodicities, and the discretizations are defined by the underlying grid.

We only consider regular, orthogonal grids, which are constructed from orthogonal
coordinate systems with equidistant discretizations along each axis. The dimension of
the space that the grid describes is given by the attribute :attr:`dim`. Points given in
these coordinates can be mapped to coordinates in Cartesian space using the methods
:meth:`point_to_cartesian` and its inverse. Moreover, points can be mapped to cell
indices using the methods :meth:`point_to_cell`.

.. autosummary::
   :nosignatures:

   ~cartesian.UnitGrid
   ~cartesian.CartesianGrid
   ~spherical.PolarSymGrid
   ~spherical.SphericalSymGrid
   ~cylindrical.CylindricalSymGrid
   
Inheritance structure of the classes:

.. inheritance-diagram:: pde.grids.base pde.grids.cartesian
        pde.grids.spherical.PolarSymGrid pde.grids.spherical.SphericalSymGrid
        pde.grids.cylindrical.CylindricalSymGrid
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from . import operators  # import all operator modules to register them
from .base import registered_operators
from .boundaries import *
from .cartesian import CartesianGrid, UnitGrid
from .cylindrical import CylindricalGrid, CylindricalSymGrid
from .spherical import PolarGrid, PolarSymGrid, SphericalGrid, SphericalSymGrid

del operators  # remove the name from the namespace
