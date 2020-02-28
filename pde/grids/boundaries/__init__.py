r"""
This package contains classes for handling different boundary conditions.
Currently supported boundary conditions are 'periodic' (for straight dimension
like in Cartesian coordinates), 'value' (where the value at the boundary is
given), and 'derivative' (where the derivative is specified).

Derivatives are given relative to the outward normal vector, such that positive
derivatives correspond to a function that increases across the boundary, which
corresponds to an inwards flux. Conversely, negative derivatives are associated
with effluxes.

Local boundary conditions:

* :class:`~pde.grids.boundaries.local.DirichletBC`:
  Imposing the value of a field at the boundary
* :class:`~pde.grids.boundaries.local.NeumannBC`:
  Imposing the derivative of a field in the outward normal direction at the
  boundary
* :class:`~pde.grids.boundaries.local.MixedBC`:
  Imposing the derivative of a field in the outward normal direction
  proportional to its value at the boundary  
* :class:`~pde.grids.boundaries.local.CurvatureBC`:
  Imposing the second derivative (curvature) of a field at the boundary
* :class:`~pde.grids.boundaries.local.ExtrapolateBC`:
  Extrapolate boundary points linearly from the two points closest to the
  boundary

Boundaries for an axis:

* :class:`~pde.grids.boundaries.axis.BoundaryPair`:
  Uses the local boundary conditions to specify the two boundaries along an
  axis.
* :class:`~pde.grids.boundaries.axis.BoundaryPeriodic`:
  Indicates that an axis has periodic boundary conditions

Boundaries for all axes of a grid:

* :class:`~pde.grids.boundaries.axes.Boundaries`:
  Collection of boundaries to describe conditions for all axes

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .local import DomainError, PeriodicityError
from .axes import Boundaries
