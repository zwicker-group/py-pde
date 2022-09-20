r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module handles the boundaries of a single axis of a grid. There are
generally only two options, depending on whether the axis of the underlying
grid is defined as periodic or not. If it is periodic, the class 
:class:`~pde.grids.boundaries.axis.BoundaryPeriodic` should be used, while
non-periodic axes have more option, which are represented by
:class:`~pde.grids.boundaries.axis.BoundaryPair`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np
from numba.extending import register_jitable

from ...tools.typing import GhostCellSetter
from ..base import GridBase, PeriodicityError
from .local import _MPIBC, BCBase, BCDataError, BoundaryData, _PeriodicBC

if TYPE_CHECKING:
    from .._mesh import GridMesh  # @UnusedImport

BoundaryPairData = Union[
    Dict[str, BoundaryData],
    BoundaryData,
    Tuple[BoundaryData, BoundaryData],
    "BoundaryAxisBase",
]


class BoundaryAxisBase:
    """base class for defining boundaries of a single axis in a grid"""

    low: BCBase
    """:class:`~pde.grids.boundaries.local.BCBase`: Boundary condition at lower end """
    high: BCBase
    """:class:`~pde.grids.boundaries.local.BCBase`: Boundary condition at upper end """

    def __init__(self, low: BCBase, high: BCBase):
        """
        Args:
            low (:class:`~pde.grids.boundaries.local.BCBase`):
                Instance describing the lower boundary
            high (:class:`~pde.grids.boundaries.local.BCBase`):
                Instance describing the upper boundary
        """
        # check data consistency
        assert low.grid == high.grid
        assert low.axis == high.axis
        assert low.rank == high.rank
        assert high.upper and not low.upper

        # check consistency with the grid
        if not (
            low.grid.periodic[low.axis]
            == isinstance(low, _PeriodicBC)
            == isinstance(high, _PeriodicBC)
        ):
            raise PeriodicityError("Periodicity of conditions must match grid")

        self.low = low
        self.high = high

    def __repr__(self):
        return f"{self.__class__.__name__}({self.low!r}, {self.high!r})"

    def __str__(self):
        if self.low == self.high:
            return str(self.low)
        else:
            return f"({self.low}, {self.high})"

    @classmethod
    def get_help(cls) -> str:
        """Return information on how boundary conditions can be set"""
        return (
            "Boundary conditions for each side can be set using a tuple: "
            f"(lower_bc, upper_bc). {BCBase.get_help()}"
        )

    def copy(self) -> BoundaryAxisBase:
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(self.low.copy(), self.high.copy())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.low == other.low
            and self.high == other.high
        )

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ != other.__class__
            or self.low != other.low
            or self.high != other.high
        )

    def __iter__(self):
        yield self.low
        yield self.high

    def __getitem__(self, index) -> BCBase:
        """returns one of the sides"""
        if index == 0 or index is False:
            return self.low
        elif index == 1 or index is True:
            return self.high
        else:
            raise IndexError("Index must be 0/False or 1/True")

    def __setitem__(self, index, data) -> None:
        """set one of the sides"""
        # determine which side was selected
        upper = {0: False, False: False, 1: True, True: True}[index]

        # create the appropriate boundary condition
        bc = BCBase.from_data(
            self.grid,
            self.axis,
            upper=upper,
            data=data,
            rank=self[upper].rank,
        )

        # set the data
        if upper:
            self.high = bc
        else:
            self.low = bc

    @property
    def grid(self) -> GridBase:
        """:class:`~pde.grids.base.GridBase`: Underlying grid"""
        return self.low.grid

    @property
    def axis(self) -> int:
        """int: The axis along which the boundaries are defined"""
        return self.low.axis

    @property
    def periodic(self) -> bool:
        """bool: whether the axis is periodic"""
        return self.grid.periodic[self.axis]

    @property
    def rank(self) -> int:
        """int: rank of the associated boundary condition"""
        assert self.low.rank == self.high.rank
        return self.low.rank

    def get_mathematical_representation(self, field_name: str = "C") -> Tuple[str, str]:
        """return mathematical representation of the boundary condition"""
        return (
            self.low.get_mathematical_representation(field_name),
            self.high.get_mathematical_representation(field_name),
        )

    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        """sets the elements of the sparse representation of this condition

        Args:
            idx (tuple):
                The index of the point that must lie on the boundary condition

        Returns:
            float, dict: A constant value and a dictionary with indices and
            factors that can be used to calculate this virtual point
        """
        axis_coord = idx[self.axis]
        if axis_coord == -1:
            # the virtual point on the lower side
            return self.low.get_data(idx)
        elif axis_coord == self.grid.shape[self.axis]:
            # the virtual point on the upper side
            return self.high.get_data(idx)
        else:
            # the normal case of an interior point
            return 0, {axis_coord: 1}

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for all boundaries

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        # send boundary information to other nodes if using MPI
        if isinstance(self.low, _MPIBC):
            self.low.send_ghost_cells(data_full, args=args)
        if isinstance(self.high, _MPIBC):
            self.high.send_ghost_cells(data_full, args=args)
        # set the actual ghost cells
        self.high.set_ghost_cells(data_full, args=args)
        self.low.set_ghost_cells(data_full, args=args)

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells for this axis on a full array"""
        # get the functions that handle the data
        ghost_cell_sender_low = self.low.make_ghost_cell_sender()
        ghost_cell_sender_high = self.high.make_ghost_cell_sender()
        ghost_cell_setter_low = self.low.make_ghost_cell_setter()
        ghost_cell_setter_high = self.high.make_ghost_cell_setter()

        @register_jitable
        def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
            """helper function setting the conditions on all axes"""
            # send boundary information to other nodes if using MPI
            ghost_cell_sender_low(data_full, args=args)
            ghost_cell_sender_high(data_full, args=args)
            # set the actual ghost cells
            ghost_cell_setter_high(data_full, args=args)
            ghost_cell_setter_low(data_full, args=args)

        return ghost_cell_setter  # type: ignore


class BoundaryPair(BoundaryAxisBase):
    """represents the two boundaries of an axis along a single dimension"""

    @classmethod
    def from_data(cls, grid: GridBase, axis: int, data, rank: int = 0) -> BoundaryPair:
        """create boundary pair from some data

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            data (str or dict):
                Data that describes the boundary pair
            rank (int):
                The tensorial rank of the field for this boundary condition

        Returns:
            :class:`~pde.grids.boundaries.axis.BoundaryPair`:
            the instance created from the data

        Throws:
            ValueError if `data` cannot be interpreted as a boundary pair
        """
        # handle the simple cases
        if isinstance(data, dict):
            if "low" in data or "high" in data:
                # separate conditions for low and high
                data_copy = data.copy()
                d_low = data_copy.pop("low")
                low = BCBase.from_data(grid, axis, upper=False, data=d_low, rank=rank)
                d_high = data_copy.pop("high")
                high = BCBase.from_data(grid, axis, upper=True, data=d_high, rank=rank)
                if data_copy:
                    raise BCDataError(f"Data items {data_copy.keys()} were not used.")
            else:
                # one condition for both sides
                low = BCBase.from_data(grid, axis, upper=False, data=data, rank=rank)
                high = BCBase.from_data(grid, axis, upper=True, data=data, rank=rank)

        elif isinstance(data, (str, BCBase)):
            # a type for both boundaries
            low = BCBase.from_data(grid, axis, upper=False, data=data, rank=rank)
            high = BCBase.from_data(grid, axis, upper=True, data=data, rank=rank)

        else:
            # the only remaining valid format is a list of conditions for the
            # lower and upper boundary
            try:
                # try obtaining the length
                data_len = len(data)
            except TypeError:
                # if len is not supported, the format must be wrong
                raise BCDataError(
                    f"Unsupported boundary format: `{data}`. " + cls.get_help()
                )
            else:
                if data_len == 2:
                    # assume that data is given for each boundary
                    low = BCBase.from_data(
                        grid, axis, upper=False, data=data[0], rank=rank
                    )
                    high = BCBase.from_data(
                        grid, axis, upper=True, data=data[1], rank=rank
                    )
                else:
                    # if the length is strange, the format must be wrong
                    raise BCDataError(
                        "Expected two conditions for the two sides of the axis, but "
                        f"got `{data}`. " + cls.get_help()
                    )

        return cls(low, high)

    def check_value_rank(self, rank: int) -> None:
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition

        Throws:
            RuntimeError: if the value does not have rank `rank`
        """
        self.low.check_value_rank(rank)
        self.high.check_value_rank(rank)


class BoundaryPeriodic(BoundaryPair):
    """represent a periodic axis"""

    def __init__(self, grid: GridBase, axis: int, flip_sign: bool = False):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            flip_sign (bool):
                Impose different signs on the two sides of the boundary
        """
        low = _PeriodicBC(grid=grid, axis=axis, upper=False, flip_sign=flip_sign)
        high = _PeriodicBC(grid=grid, axis=axis, upper=True, flip_sign=flip_sign)
        super().__init__(low, high)

    @property
    def flip_sign(self):
        """bool: Whether different signs are imposed on the two sides of the boundary"""
        return self.low.flip_sign

    def __repr__(self):
        res = f"{self.__class__.__name__}(grid={self.grid}, axis={self.axis}"
        if self.flip_sign:
            return res + ", flip_sign=True)"
        else:
            return res + ")"

    def __str__(self):
        if self.flip_sign:
            return '"anti-periodic"'
        else:
            return '"periodic"'

    def copy(self) -> BoundaryPeriodic:
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(grid=self.grid, axis=self.axis, flip_sign=self.flip_sign)

    def check_value_rank(self, rank: int) -> None:
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition
        """


def get_boundary_axis(
    grid: GridBase, axis: int, data, rank: int = 0
) -> BoundaryAxisBase:
    """return object representing the boundary condition for a single axis

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid for which the boundary conditions are defined
        axis (int):
            The axis to which this boundary condition is associated
        data (str or tuple or dict):
            Data describing the boundary conditions for this axis
        rank (int):
            The tensorial rank of the field for this boundary condition

    Returns:
        :class:`~pde.grids.boundaries.axis.BoundaryAxisBase`:
            Appropriate boundary condition for the axis
    """
    # handle special constructs that describe boundary conditions
    if (
        data == "natural"
        or data == "auto_periodic_neumann"
        or data == "auto_periodic_derivative"
    ):
        # automatic choice between periodic and Neumann condition
        if grid.periodic[axis]:
            data = "periodic"
        else:
            data = "derivative"

    elif data == "auto_periodic_dirichlet" or data == "auto_periodic_value":
        # automatic choice between periodic and Dirichlet condition
        if grid.periodic[axis]:
            data = "periodic"
        else:
            data = "value"

    # handle different types of data that specify boundary conditions
    if isinstance(data, BoundaryAxisBase):
        # boundary is already an the correct format
        bcs = data
    elif data == "periodic" or data == ("periodic", "periodic"):
        # initialize a periodic boundary condition
        bcs = BoundaryPeriodic(grid, axis)
    elif data == "anti-periodic" or data == ("anti-periodic", "anti-periodic"):
        # initialize a anti-periodic boundary condition
        bcs = BoundaryPeriodic(grid, axis, flip_sign=True)
    elif isinstance(data, dict) and data.get("type") == "periodic":
        # initialize a periodic boundary condition
        bcs = BoundaryPeriodic(grid, axis)
    else:
        # initialize independent boundary conditions for the two sides
        bcs = BoundaryPair.from_data(grid, axis, data, rank=rank)

    # check consistency
    if bcs.periodic != grid.periodic[axis]:
        raise PeriodicityError("Periodicity of conditions must match grid")
    return bcs
