r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module handles the boundaries of all axes of a grid. It only defines
:class:`Boundaries`, which acts as a list of
:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
from numba.extending import register_jitable

from ...tools.typing import GhostCellSetter
from ..base import GridBase, PeriodicityError
from .axis import BoundaryPair, BoundaryPairData, get_boundary_axis
from .local import BCDataError

BoundariesData = Union[BoundaryPairData, Sequence[BoundaryPairData]]


class Boundaries(list):
    """class that bundles all boundary conditions for all axes"""

    grid: GridBase
    """:class:`~pde.grids.base.GridBase`: grid for which boundaries are defined """

    def __init__(self, boundaries):
        """initialize with a list of boundaries"""
        if len(boundaries) == 0:
            raise BCDataError("List of boundaries must not be empty")

        # extract grid
        self.grid = boundaries[0].grid

        # check dimension
        if len(boundaries) != self.grid.num_axes:
            raise BCDataError(f"Need boundary conditions for {self.grid.num_axes} axes")

        # check consistency
        for axis, boundary in enumerate(boundaries):
            if boundary.grid != self.grid:
                raise BCDataError("Boundaries are not defined on the same grid")
            if boundary.axis != axis:
                raise BCDataError(
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
    def from_data(cls, grid: GridBase, boundaries, rank: int = 0) -> Boundaries:
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
                The tensorial rank of the field for this boundary condition
        """
        # check whether this is already the correct class
        if isinstance(boundaries, Boundaries):
            # boundaries are already in the correct format
            assert boundaries.grid == grid
            boundaries.check_value_rank(rank)
            return boundaries

        # convert natural boundary conditions if present
        if (
            boundaries == "natural"
            or boundaries == "auto_periodic_neumann"
            or boundaries == "auto_periodic_derivative"
        ):
            # set the respective natural conditions for all axes
            boundaries = []
            for periodic in grid.periodic:
                if periodic:
                    boundaries.append("periodic")
                else:
                    boundaries.append("derivative")

        elif (
            boundaries == "auto_periodic_dirichlet"
            or boundaries == "auto_periodic_value"
        ):
            # set the respective natural conditions (with vanishing values) for all axes
            boundaries = []
            for periodic in grid.periodic:
                if periodic:
                    boundaries.append("periodic")
                else:
                    boundaries.append("value")

        elif boundaries == "auto_periodic_curvature":
            # set the respective natural conditions (with vanishing curvature) for all axes
            boundaries = []
            for periodic in grid.periodic:
                if periodic:
                    boundaries.append("periodic")
                else:
                    boundaries.append("curvature")

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
            raise BCDataError(
                f"Unsupported boundary format: `{boundaries}`. " + cls.get_help()
            )

        return cls(bcs)

    def __eq__(self, other):
        if not isinstance(other, Boundaries):
            return NotImplemented
        return super().__eq__(other) and self.grid == other.grid

    def check_value_rank(self, rank: int) -> None:
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition

        Throws:
            RuntimeError: if any value does not have rank `rank`
        """
        for b in self:
            b.check_value_rank(rank)

    @classmethod
    def get_help(cls) -> str:
        """Return information on how boundary conditions can be set"""
        return (
            "Boundary conditions for each axis are set using a list: [bc_x, bc_y, "
            "bc_z]. If the associated axis is periodic, the boundary condition needs "
            f"to be set to 'periodic'. Otherwise, {BoundaryPair.get_help()}"
        )

    def copy(self) -> Boundaries:
        """create a copy of the current boundaries"""
        return self.__class__([bc.copy() for bc in self])

    @property
    def periodic(self) -> List[bool]:
        """:class:`~numpy.ndarray`: a boolean array indicating which dimensions
        are periodic according to the boundary conditions"""
        return self.grid.periodic

    def __getitem__(self, index):
        """extract specific boundary conditions

        Args:
            index (int or str):
                Index can either be a number or an axes name, indicating the axes of
                which conditions are returned. Alternatively, `index` can be a named
                boundary whose conditions will then be returned
        """
        if isinstance(index, str):
            # assume that the index is a known identifier
            if index in self.grid.boundary_names:
                axis, upper = self.grid.boundary_names[index]
                return self[axis][upper]
            elif index in self.grid.axes:
                return self[self.grid.axes.index(index)]
            else:
                raise KeyError(index)
        else:
            # handle all other cases, in particular integer indices
            return super().__getitem__(index)

    def __setitem__(self, index, data) -> None:
        """set specific boundary conditions

        Args:
            index (int or str):
                Index can either be a number or an axes name, indicating the axes of
                which conditions are returned. Alternatively, `index` can be a named
                boundary whose conditions will then be returned
            data:
                Data describing the boundary conditions for this axis or side
        """
        if isinstance(index, str):
            # assume that the index is a known identifier
            if index in self.grid.boundary_names:
                # set a specific boundary side
                axis, upper = self.grid.boundary_names[index]
                self[axis][upper] = data

            elif index in self.grid.axes:
                # set both sides of the boundary condition
                axis = self.grid.axes.index(index)
                self[axis] = get_boundary_axis(
                    grid=self.grid,
                    axis=axis,
                    data=data,
                    rank=self[axis].rank,
                )
            else:
                raise KeyError(index)

        else:
            # handle all other cases, in particular integer indices
            return super().__setitem__(index, data)

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        result = []
        for b in self:
            try:
                result.extend(b.get_mathematical_representation(field_name))
            except NotImplementedError:
                axis_name = self.grid.axes[b.axis]
                result.append(f"Representation not implemented for axis {axis_name}")

        return "\n".join(result)

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cells for all boundaries

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        for b in self:
            b.set_ghost_cells(data_full, args=args)

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells on a full array"""
        ghost_cell_setters = tuple(b.make_ghost_cell_setter() for b in self)

        # TODO: use numba.literal_unroll
        # # get the setters for all axes
        #
        # from pde.tools.numba import jit
        #
        # @jit
        # def set_ghost_cells(data_full: np.ndarray, args=None) -> None:
        #     for f in nb.literal_unroll(ghost_cell_setters):
        #         f(data_full, args=args)
        #
        # return set_ghost_cells

        def chain(
            fs: Sequence[GhostCellSetter], inner: GhostCellSetter = None
        ) -> GhostCellSetter:
            """helper function composing setters of all axes recursively"""

            first, rest = fs[0], fs[1:]

            if inner is None:

                @register_jitable
                def wrap(data_full: np.ndarray, args=None) -> None:
                    first(data_full, args=args)

            else:

                @register_jitable
                def wrap(data_full: np.ndarray, args=None) -> None:
                    inner(data_full, args=args)  # type: ignore
                    first(data_full, args=args)

            if rest:
                return chain(rest, wrap)
            else:
                return wrap  # type: ignore

        return chain(ghost_cell_setters)
