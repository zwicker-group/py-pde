'''
Grids define the domains on which the pde will be solved. In particular, 
symmetries, periodicities, and the discretizations are also defined here.

We only consider regular, orthogonal grids, which are constructed from
orthogonal coordinate systems with equidistant discretizations along each axis.
The dimension of the space that the grid describes is given by the attribute
:attr:`dim`. Points given in these coordinates can be mapped to coordinates in
Cartesian space using the methods :meth:`point_to_cartesian` and its inverse.
Moreover, points can be mapped to cell indices using the methods
:meth:`point_to_cell`.

.. autosummary::
   :nosignatures:

   ~cartesian.UnitGrid
   ~cartesian.CartesianGrid
   ~spherical.PolarGrid
   ~spherical.SphericalGrid
   ~cylindrical.CylindricalGrid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
'''

from .cartesian import UnitGrid, CartesianGrid
from .spherical import PolarGrid, SphericalGrid
from .cylindrical import CylindricalGrid
