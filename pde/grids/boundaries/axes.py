r"""This module handles the boundaries of all axes of a grid.

.. autosummary::
   :nosignatures:

   ~BoundariesBase
   ~BoundariesList
   ~BoundariesSetter

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator, Sequence
from typing import Union

import numpy as np
from numba.extending import register_jitable

from ...tools.numba import jit
from ...tools.typing import GhostCellSetter
from ..base import GridBase, PeriodicityError
from .axis import BoundaryAxisBase, BoundaryPair, BoundaryPairData, get_boundary_axis
from .local import BCBase, BCDataError

BoundariesData = Union[BoundaryPairData, Sequence[BoundaryPairData]]


class BoundariesBase:
    """Base class keeping information about how to set conditions on all boundaries."""

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """Set the ghost cells for all boundaries.

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            set_corners (bool):
                Determines whether the corner cells are set using interpolation
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        raise NotImplementedError

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """Return function that sets the ghost cells on a full array.

        Returns:
            Callable with signature :code:`(data_full: np.ndarray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        raise NotImplementedError

    @classmethod
    def from_data(cls, data, **kwargs) -> BoundariesBase:
        r"""Creates all boundaries from given data.

        Args:
            data (str or list or tuple or dict or callable):
                Data that describes the boundaries. If this is a callable, we create
                :class:`~pde.grids.boundaries.axes.BoundariesSetter`. In all other,
                cases :class:`~pde.grids.boundaries.axes.BoundariesList` is created and
                `data` can either be a list of specifications for each dimension or a
                single one, which is then applied to all dimensions. The boundary for a
                dimensions can be specified by one of the following formats:

                * string specifying a single type for all boundaries
                * dictionary specifying the type and values for all boundaries
                * tuple pair specifying the low and high boundary individually
            **kwargs:
                In some cases additional data can be specified or is even required. For
                instance, :class:`~pde.grids.boundaries.axes.BoundariesList` expects
                a `grid` (:class:`~pde.grids.base.GridBase`): to which the boundary
                condition are associated, and it can use a `rank` (int), which sets
                the tensorial rank of the field for this boundary condition.
        """
        # check whether this is already of the correct type
        if isinstance(data, BoundariesBase):
            return data.__class__.from_data(data, **kwargs)

        # best guess based on the data:
        if callable(data):
            return BoundariesSetter.from_data(data)
        else:
            return BoundariesList.from_data(data, **kwargs)

    @classmethod
    def get_help(cls) -> str:
        """Return information on how boundary conditions can be set."""
        return (
            "Boundary conditions for each axis are set using a list: [bc_x, bc_y, "
            "bc_z]. If the associated axis is periodic, the boundary condition needs "
            f"to be set to 'periodic'. Otherwise, {BoundaryPair.get_help()}"
        )


class BoundariesList(BoundariesBase):
    """Defines boundary conditions for all axes individually."""

    def __init__(self, boundaries: list[BoundaryAxisBase]):
        """Initialize with a list of boundaries."""
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
                raise BCDataError("BoundariesList are not defined on the same grid")
            if boundary.axis != axis:
                raise BCDataError(
                    "BoundariesList need to be ordered like the respective axes"
                )
            if boundary.periodic != self.grid.periodic[axis]:
                raise PeriodicityError(
                    "Periodicity specified in the boundaries conditions is not "
                    f"compatible with the grid ({boundary.periodic} != "
                    f"{self.grid.periodic[axis]} for axis {axis})"
                )

        # create the list of boundaries
        self._axes: list[BoundaryAxisBase] = boundaries

    @classmethod
    def from_data(  # type: ignore
        cls, data, *, grid: GridBase, rank: int = 0, **kwargs
    ) -> BoundariesList:
        """Creates all boundaries from given data.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid with which the boundary condition is associated
            data (str or list or tuple or dict):
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
        if isinstance(data, BoundariesList):
            # boundaries are already in the correct format
            if data.grid._mesh is not None:
                # we need to exclude this case since otherwise we get into a rabbit hole
                # where it is not clear what grid boundary conditions belong to. The
                # idea is that users only create boundary conditions for the full grid
                # and that the splitting onto subgrids is only done once, automatically,
                # and without involving calls to `from_data`
                raise ValueError("Cannot create MPI subgrid BC from data")

            if data.grid != grid:
                raise ValueError(
                    "The grid of the supplied boundary condition is incompatible with "
                    f"the current grid ({data.grid!r} != {grid!r})"
                )
            data.check_value_rank(rank)
            return data

        elif isinstance(data, BoundariesBase):
            raise TypeError(
                "Can only use type `BoundariesList`. Use `BoundariesBase.from_data` "
                "for more general data."
            )

        # convert natural boundary conditions if present
        if (
            data == "natural"
            or data == "auto_periodic_neumann"
            or data == "auto_periodic_derivative"
        ):
            # set the respective natural conditions for all axes
            data = []
            for periodic in grid.periodic:
                if periodic:
                    data.append("periodic")
                else:
                    data.append("derivative")

        elif data == "auto_periodic_dirichlet" or data == "auto_periodic_value":
            # set the respective natural conditions (with vanishing values) for all axes
            data = []
            for periodic in grid.periodic:
                if periodic:
                    data.append("periodic")
                else:
                    data.append("value")

        elif data == "auto_periodic_curvature":
            # set the respective natural conditions (with vanishing curvature) for all axes
            data = []
            for periodic in grid.periodic:
                if periodic:
                    data.append("periodic")
                else:
                    data.append("curvature")

        # create the list of BoundaryAxis objects
        bcs = None
        if isinstance(data, (str, dict)):
            # one specification for all axes
            bcs = [
                get_boundary_axis(grid, i, data, rank=rank)
                for i in range(grid.num_axes)
            ]

        elif hasattr(data, "__len__"):
            # handle cases that look like sequences
            if len(data) == grid.num_axes:
                # assume that data is given for each boundary
                bcs = [
                    get_boundary_axis(grid, i, boundary, rank=rank)
                    for i, boundary in enumerate(data)
                ]
            elif grid.num_axes == 1 and len(data) == 2:
                # special case where the two sides can be specified directly
                bcs = [get_boundary_axis(grid, 0, data, rank=rank)]

        if bcs is None:
            # none of the logic worked
            raise BCDataError(
                f"Unsupported boundary format: `{data}`. " + cls.get_help()
            )

        return BoundariesList(bcs)

    def __str__(self):
        items = ", ".join(str(item) for item in self)
        return f"[{items}]"

    def __len__(self):
        return len(self._axes)

    def __iter__(self) -> Iterator[BoundaryAxisBase]:
        yield from self._axes

    def __eq__(self, other):
        if not isinstance(other, BoundariesList):
            return NotImplemented
        return self.grid == other.grid and self._axes == other._axes

    def __ne__(self, other):
        if not isinstance(other, BoundariesList):
            return NotImplemented
        return self.grid != other.grid or self._axes != other._axes

    @property
    def boundaries(self) -> Iterator[BCBase]:
        """Iterator over all non-periodic boundaries."""
        for boundary_axis in self._axes:  # iterate all axes
            if not boundary_axis.periodic:  # skip periodic axes
                yield from boundary_axis

    def check_value_rank(self, rank: int) -> None:
        """Check whether the values at the boundaries have the correct rank.

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition

        Throws:
            RuntimeError: if any value does not have rank `rank`
        """
        for b in self._axes:
            b.check_value_rank(rank)

    def copy(self) -> BoundariesList:
        """Create a copy of the current boundaries."""
        return self.__class__([bc.copy() for bc in self._axes])

    @property
    def periodic(self) -> list[bool]:
        """:class:`~numpy.ndarray`: a boolean array indicating which dimensions are
        periodic according to the boundary conditions."""
        return self.grid.periodic

    def __getitem__(self, index):
        """Extract specific boundary conditions.

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
                return self._axes[axis][upper]
            elif index in self.grid.axes:
                return self._axes[self.grid.axes.index(index)]
            else:
                raise KeyError(index)
        else:
            # handle all other cases, in particular integer indices
            return self._axes[index]

    def __setitem__(self, index, data) -> None:
        """Set specific boundary conditions.

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
                self._axes[axis][upper] = data

            elif index in self.grid.axes:
                # set both sides of the boundary condition
                axis = self.grid.axes.index(index)
                self._axes[axis] = get_boundary_axis(
                    grid=self.grid,
                    axis=axis,
                    data=data,
                    rank=self[axis].rank,
                )
            else:
                raise KeyError(index)

        else:
            # handle all other cases, in particular integer indices
            self._axes[index] = data

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        result: list[str] = []
        for b in self._axes:
            try:
                result.extend(b.get_mathematical_representation(field_name))
            except NotImplementedError:
                axis_name = self.grid.axes[b.axis]
                result.append(f"Representation not implemented for axis {axis_name}")

        return "\n".join(result)

    def set_ghost_cells(
        self, data_full: np.ndarray, *, set_corners: bool = False, args=None
    ) -> None:
        """Set the ghost cells for all boundaries.

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            set_corners (bool):
                Determines whether the corner cells are set using interpolation
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        for b in self:
            b.set_ghost_cells(data_full, args=args)

        if set_corners and self.grid.num_axes >= 2:
            d = data_full  # abbreviation
            nxt = [1, -2]  # maps 0 to 1 and -1 to -2 to obtain neighboring cells
            if self.grid.num_axes == 2:
                # iterate all corners
                for i, j in itertools.product([0, -1], [0, -1]):
                    d[..., i, j] = (d[..., nxt[i], j] + d[..., i, nxt[j]]) / 2

            elif self.grid.num_axes == 3:
                # iterate all edges
                for i, j in itertools.product([0, -1], [0, -1]):
                    d[..., :, i, j] = (+d[..., :, nxt[i], j] + d[..., :, i, nxt[j]]) / 2
                    d[..., i, :, j] = (+d[..., nxt[i], :, j] + d[..., i, :, nxt[j]]) / 2
                    d[..., i, j, :] = (+d[..., nxt[i], j, :] + d[..., i, nxt[j], :]) / 2
                # iterate all corners
                for i, j, k in itertools.product(*[[0, -1]] * 3):
                    d[..., i, j, k] = (
                        d[..., nxt[i], j, k]
                        + d[..., i, nxt[j], k]
                        + d[..., i, j, nxt[k]]
                    ) / 3

            elif self.grid.num_axes > 3:
                raise NotImplementedError(
                    f"Can't interpolate corners for grid with {self.grid.num_axes} axes"
                )

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """Return function that sets the ghost cells on a full array."""
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
            fs: Sequence[GhostCellSetter], inner: GhostCellSetter | None = None
        ) -> GhostCellSetter:
            """Helper function composing setters of all axes recursively."""

            first, rest = fs[0], fs[1:]

            if inner is None:

                @register_jitable
                def wrap(data_full: np.ndarray, args=None) -> None:
                    first(data_full, args=args)

            else:

                @register_jitable
                def wrap(data_full: np.ndarray, args=None) -> None:
                    inner(data_full, args=args)
                    first(data_full, args=args)

            if rest:
                return chain(rest, wrap)
            else:
                return wrap  # type: ignore

        return chain(ghost_cell_setters)


class BoundariesSetter(BoundariesBase):
    """Represents a function that sets ghost cells to determine boundary conditions.

    The function must have accept a :class:`~numpy.ndarray`, which contains the full
    field data including the ghost points, and a second, optional argument, which is a
    dictionary containing additional parameters, like the current time point `t` in case
    of a simulation.

    Example:
        Here is an example for a simple boundary setter, which sets specific boundary
        conditions in the x-direction and periodic conditions in the y-direction of a
        grid with two axes. Note that this boundary condition will not work for grids
        with other number of axes and no additional checks are performed.

        .. code-block:: python

            def setter(data, args=None):
                data[0, :] = data[1, :]  # Vanishing derivative at left side
                data[-1, :] = 2 - data[-2, :]  # Fixed value `1` at right side
                data[:, 0] = data[:, -2]  # Periodic BC at top
                data[:, -1] = data[:, 1]  # Periodic BC at bottom
    """

    def __init__(self, setter: GhostCellSetter):
        self._setter = setter

    @classmethod
    def from_data(cls, data, **kwargs) -> BoundariesSetter:
        """Creates all boundaries from given data.

        Args:
            data (callable):
                Function that sets the ghost cells
        """
        # check whether this is already the correct class
        if isinstance(data, BoundariesSetter):
            # boundaries are already in the correct format
            return data

        elif isinstance(data, BoundariesBase):
            raise TypeError(
                "Can only use type `BoundariesSetter`. Use `BoundariesBase.from_data` "
                "for more general data."
            )

        return BoundariesSetter(data)

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """Set the ghost cells for all boundaries.

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            set_corners (bool):
                Determines whether the corner cells are set using interpolation
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        self._setter(data_full, args=args)

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """Return function that sets the ghost cells on a full array.

        Returns:
            Callable with signature :code:`(data_full: np.ndarray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        return jit(self._setter)  # type: ignore
