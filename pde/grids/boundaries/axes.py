r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module handles the boundaries of all axes of a grid. It only defines
:class:`Boundaries`, which acts as a list of
:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`.
"""

from typing import Sequence, Union

import numpy as np

from ..base import GridBase, PeriodicityError
from .axis import BoundaryPair, BoundaryPairData, get_boundary_axis

BoundariesData = Union[BoundaryPairData, Sequence[BoundaryPairData]]


class Boundaries(list):
    """ class that bundles all boundary conditions for all axes """

    grid: GridBase
    """ :class:`~pde.grids.base.GridBase`:
    The grid for which the boundaries are defined """

    def __init__(self, boundaries):
        """ initialize with a list of boundaries """
        if len(boundaries) == 0:
            raise ValueError("List of boundaries must not be empty")

        # extract grid
        self.grid = boundaries[0].grid

        # check dimension
        if len(boundaries) != self.grid.num_axes:
            raise ValueError(f"Need boundary conditions for {self.grid.num_axes} axes")
        # check consistency
        for axis, boundary in enumerate(boundaries):
            if boundary.grid != self.grid:
                raise ValueError("Boundaries are not defined on the same grid")
            if boundary.axis != axis:
                raise ValueError(
                    "Boundaries need to be ordered like the respective axes"
                )
            if boundary.periodic != self.grid.periodic[axis]:
                raise PeriodicityError(
                    "Periodicity specified in the boundaries conditions is not "
                    f"compatible with the grid ({boundary.periodic} != "
                    f"{self.grid.periodic[axis]} for axis {axis})"
                )

        # create the list of boundaries
        super().__init__(boundaries)

    def __str__(self):
        items = ", ".join(str(item) for item in self)
        return f"[{items}]"

    @classmethod
    def from_data(cls, grid: GridBase, boundaries, rank: int = 0) -> "Boundaries":
        """
        Creates all boundaries from given data

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid with which the boundary condition is associated
            boundaries (str or list or tuple or dict):
                Data that describes the boundaries. This can either be a list of
                specifications for each dimension or a single one, which is then
                applied to all dimensions. The boundary for a dimensions can be
                specified by one of the following formats:

                * string specifying a single type for all boundaries
                * dictionary specifying the type and values for all boundaries
                * tuple pair specifying the low and high boundary individually
            rank (int):
                The tensorial rank of the value associated with the boundary
                conditions.

        """
        # check whether this is already the correct class
        if isinstance(boundaries, Boundaries):
            # boundaries are already in the correct format
            assert boundaries.grid == grid
            boundaries.check_value_rank(rank)
            return boundaries

        # convert natural boundary conditions if present
        if boundaries == "natural" or boundaries == "auto_periodic_neumann":
            # set the respective natural conditions for all axes
            boundaries = [
                "periodic" if periodic else "neumann" for periodic in grid.periodic
            ]
        elif boundaries == "auto_periodic_dirichlet":
            # set the respective natural conditions (with vanishing values) for all axes
            boundaries = [
                "periodic" if periodic else "dirichlet" for periodic in grid.periodic
            ]

        # create the list of BoundaryAxis objects
        bcs = None
        if isinstance(boundaries, (str, dict)):
            # one specification for all axes
            bcs = [
                get_boundary_axis(grid, i, boundaries, rank=rank)
                for i in range(grid.num_axes)
            ]

        elif hasattr(boundaries, "__len__"):
            # handle cases that look like sequences
            if len(boundaries) == grid.num_axes:
                # assume that data is given for each boundary
                bcs = [
                    get_boundary_axis(grid, i, boundary, rank=rank)
                    for i, boundary in enumerate(boundaries)
                ]
            elif grid.num_axes == 1 and len(boundaries) == 2:
                # special case where the two sides can be specified directly
                bcs = [get_boundary_axis(grid, 0, boundaries, rank=rank)]

        if bcs is None:
            # none of the logic worked
            raise ValueError(
                f"Unsupported boundary format: `{boundaries}`. " + cls.get_help()
            )

        return cls(bcs)

    def __eq__(self, other):
        if not isinstance(other, Boundaries):
            return NotImplemented
        return super().__eq__(other) and self.grid == other.grid

    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """
        return hash(tuple(bc_ax._cache_hash() for bc_ax in self))

    def check_value_rank(self, rank: int):
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (tuple): The rank of the value that is stored with this
                boundary condition

        Throws:
            RuntimeError: if any value does not have rank `rank`
        """
        for b in self:
            b.check_value_rank(rank)

    @classmethod
    def get_help(cls) -> str:
        """ Return information on how boundary conditions can be set """
        return (
            "Boundary conditions for each axis are set using a list: [bc_x, bc_y, "
            "bc_z]. If the associated axis is periodic, the boundary condition needs "
            f"to be set to 'periodic'. Otherwise, {BoundaryPair.get_help()}"
        )

    def copy(self, value=None) -> "Boundaries":
        """create a copy of the current boundaries

        Args:
            value (float or array, optional):
                If given, this changes the value stored with the boundary
                conditions. The interpretation of this value depends on the type
                of boundary condition.
            copy_grid (bool):
                Whether the grid should also be copied
        """
        result = self.__class__([bc.copy() for bc in self])
        if value is not None:
            result.set_value(value)
        return result

    @property
    def periodic(self) -> np.ndarray:
        """:class:`numpy.ndarray`: a boolean array indicating which dimensions
        are periodic according to the boundary conditions"""
        return self.grid.periodic

    def set_value(self, value=0):
        """set the value of all non-periodic boundaries

        Args:
            value (float or array):
                Sets the value stored with the boundary conditions. The
                interpretation of this value depends on the type of boundary
                condition.
        """
        for b in self:
            if not b.periodic:
                b.set_value(value)

    def scale_value(self, factor: float = 1):
        """scales the value of the boundary condition with the given factor

        Args:
            value (float):
                Scales the value associated with the boundary condition by the factor
        """
        for b in self:
            if not b.periodic:
                b.scale_value(factor)

    @property
    def _scipy_border_mode(self) -> dict:
        """dict: return a dictionary that can be used in the scipy ndimage
        functions to specify the border mode. If the current boundary cannot be
        represented by these modes, a RuntimeError is raised
        """
        mode: dict = self[0]._scipy_border_mode
        for b in self[1:]:
            if mode != b._scipy_border_mode:
                raise RuntimeError("Incompatible dimensions")
        return mode

    @property
    def _uniform_discretization(self) -> float:
        """ float: returns the uniform discretization or raises RuntimeError """
        dx_mean = np.mean(self.grid.discretization)
        if np.allclose(self.grid.discretization, dx_mean):
            return float(dx_mean)
        else:
            raise RuntimeError("Grid discretization is not uniform")

    def extract_component(self, *indices) -> "Boundaries":
        """extracts the boundary conditions of the given extract_component.

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        boundaries = [boundary.extract_component(*indices) for boundary in self]
        return self.__class__(boundaries)

    @property
    def differentiated(self) -> "Boundaries":
        """ Domain: with differentiated versions of all boundary conditions """
        return self.__class__([b.differentiated for b in self])
