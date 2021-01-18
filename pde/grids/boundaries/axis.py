r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module handles the boundaries of a single axis of a grid. There are
generally only two options, depending on whether the axis of the underlying
grid is defined as periodic or not. If it is periodic, the class 
:class:`~pde.grids.boundaries.axis.BoundaryPeriodic` should be used, while
non-periodic axes have more option, which are represented by
:class:`~pde.grids.boundaries.axis.BoundaryPair`.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np
from numba.extending import register_jitable

from ..base import DomainError, GridBase
from .local import BCBase, BoundaryData, NeumannBC, _make_get_arr_1d

BoundaryPairData = Union[
    Dict[str, BoundaryData], BoundaryData, Tuple[BoundaryData, BoundaryData]
]


class BoundaryAxisBase:
    """ base class for defining boundaries of a single axis in a grid """

    grid: GridBase
    """ :class:`~pde.grids.base.GridBase`:
    The grid for which the boundaries are defined """
    axis: int
    """ int: The axis along which the boundaries are defined """


class BoundaryPair(BoundaryAxisBase):
    """ represents the two boundaries of an axis along a single dimension """

    periodic = False

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

        self.low = low
        self.high = high
        self.grid = low.grid
        self.axis = low.axis

    def __iter__(self):
        yield self.low
        yield self.high

    def __repr__(self):
        return f"{self.__class__.__name__}({self.low!r}, {self.high!r})"

    def __str__(self):
        if self.low == self.high:
            return str(self.low)
        else:
            return f"({self.low}, {self.high})"

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

    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """
        return hash((self.low._cache_hash(), self.high._cache_hash()))

    def copy(self) -> "BoundaryPair":
        """ return a copy of itself, but with a reference to the same grid """
        return self.__class__(self.low.copy(), self.high.copy())

    def __getitem__(self, index: Union[int, bool]) -> BCBase:
        """ returns one of the sides """
        if index == 0 or index is False:
            return self.low
        elif index == 1 or index is True:
            return self.high
        else:
            raise IndexError("Index can be either 0/False or 1/True")

    def set_value(self, value=0):
        """set the value of both boundary conditions

        Args:
            value (float or array):
                Sets the value stored with the boundary conditions. The
                interpretation of this value depends on the type of boundary
                condition.
        """
        self.low.value = value
        self.high.value = value

    def scale_value(self, factor: float = 1):
        """scales the value of the boundary condition with the given factor

        Args:
            value (float):
                Scales the value associated with the boundary condition by the factor
        """
        self.low.value = factor * self.low.value  # type: ignore
        self.high.value = factor * self.high.value  # type: ignore

    @classmethod
    def get_help(cls) -> str:
        """ Return information on how boundary conditions can be set """
        return (
            "Boundary conditions for each side can be set using a tuple: "
            f"(lower_bc, upper_bc). {BCBase.get_help()}"
        )

    @classmethod
    def from_data(
        cls, grid: GridBase, axis: int, data, rank: int = 0
    ) -> "BoundaryPair":
        """create boundary pair from some data

        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            data (str or dict):
                Data that describes the boundary pair
            rank (int):
                The tensorial rank of the value associated with the boundary
                conditions.

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
                low = BCBase.from_data(
                    grid, axis, upper=False, data=data_copy.pop("low"), rank=rank
                )
                high = BCBase.from_data(
                    grid, axis, upper=True, data=data_copy.pop("high"), rank=rank
                )
                if data_copy:
                    raise ValueError(f"Data items {data_copy.keys()} were not used.")
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
                raise ValueError(
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
                    raise ValueError(
                        "Expected two conditions for the two sides of the axis, but "
                        f"got `{data}`. " + cls.get_help()
                    )

        return cls(low, high)

    @property
    def _scipy_border_mode(self) -> dict:
        """dict: a dictionary that can be used in scipy functions

        This returns arguments that can be passed to functions of the
        scipy.ndimage module to specify border conditions.

        Raise:
            RuntimeError if the boundary cannot be represented
        """
        if self.low != self.high:
            raise RuntimeError("Incompatible boundaries")

        # check whether both sides have vanishing derivative conditions
        zero_neumann_bcs = all(
            isinstance(b, NeumannBC) and np.all(b.value == 0)
            for b in [self.low, self.high]
        )
        if zero_neumann_bcs:
            return {"mode": "reflect"}
        else:
            # BoundaryCondition.value cannot be supported since the scipy value
            # mode='constant' applies the boundary conditions at a different
            # position then we would
            raise RuntimeError("Unsupported boundaries")

    def extract_component(self, *indices):
        """extracts the boundary pair of the given index.

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        bc_sub_low = self.low.extract_component(*indices)
        bc_sub_high = self.high.extract_component(*indices)
        return self.__class__(bc_sub_low, bc_sub_high)

    def check_value_rank(self, rank: int):
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int): The rank of the value that is stored with this
                boundary condition

        Throws:
            RuntimeError: if the value does not have rank `rank`
        """
        self.low.check_value_rank(rank)
        self.high.check_value_rank(rank)

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

    def make_virtual_point_evaluators(self) -> Tuple[Callable, Callable]:
        """returns two functions evaluating the value at virtual support points

        Args:
            size (int): Number of support points along the axis
            dx (float): Discretization, i.e., distance between support points

        Returns:
            tuple: Two functions that each take a 1d array as an argument and
            return the associated value at the virtual support point outside the
            lower and upper boundary, respectively.
        """
        eval_low = self.low.make_virtual_point_evaluator()
        eval_high = self.high.make_virtual_point_evaluator()
        return (eval_low, eval_high)

    @property
    def differentiated(self) -> "BoundaryPair":
        """ BoundaryPair: differentiated version of this boundary condition """
        return self.__class__(self.low.differentiated, self.high.differentiated)

    def get_point_evaluator(self, fill: np.array = None) -> Callable:
        """return a function to evaluate values at a given point

        The point can either be a point inside the domain or a virtual point
        right outside the domain

        Args:
            fill (:class:`numpy.ndarray`, optional):
                Determines how values out of bounds are handled. If `None`, a
                `DomainError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.

        Returns:
            function: A function taking a 1d array and an index as an argument,
                returning the value of the array at this index.
        """
        size = self.low.grid.shape[self.low.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        eval_low = self.low.make_virtual_point_evaluator()
        eval_high = self.high.make_virtual_point_evaluator()

        @register_jitable
        def evaluate(arr, idx):
            """ evaluate values of the 1d array `arr_1d` at an index `i` """
            arr_1d, i, _ = get_arr_1d(arr, idx)

            if i == -1:
                # virtual point on the lower side of the axis
                return eval_low(arr, idx)

            elif i == size:
                # virtual point on the upper side of the axis
                return eval_high(arr, idx)

            elif 0 <= i < size:
                # inner point of the axis
                return arr_1d[..., i]

            elif fill is None:
                # point is outside the domain and no fill value is specified
                raise DomainError("Point index lies outside bounds")

            else:
                # Point is outside the domain, but fill value is specified. Note
                # that fill value needs to be given with the correct shape.
                return fill

        return evaluate  # type: ignore

    def make_region_evaluator(self) -> Callable:
        """return a function to evaluate values in a neighborhood of a point

        Returns:
            function: A function that can be called with the data array and a
            tuple indicating around what point the region is evaluated. The
            function returns the data values left of the point, at the point,
            and right of the point along the axis associated with this boundary
            condition. The function takes boundary conditions into account if
            the point lies on the boundary.
        """
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        ap_low = self.low.make_adjacent_evaluator()
        ap_high = self.high.make_adjacent_evaluator()

        @register_jitable
        def region_evaluator(arr, idx: Tuple[int, ...]) -> Tuple[float, float, float]:
            """ compiled function return the values in the region """
            # extract the 1d array along axis
            arr_1d, i_point, bc_idx = get_arr_1d(arr, idx)
            return (
                ap_low(arr_1d, i_point, bc_idx),
                arr_1d[..., i_point],
                ap_high(arr_1d, i_point, bc_idx),
            )

        return region_evaluator  # type: ignore


class BoundaryPeriodic(BoundaryAxisBase):
    """ represent a periodic axis """

    periodic = True
    _scipy_border_mode = {"mode": "wrap"}

    def __init__(self, grid: GridBase, axis: int):
        """
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
        """
        self.grid = grid
        self.axis = axis

    def __iter__(self):
        return iter(())  # there are no sub-boundaries to iterate over

    def __repr__(self):
        return f"{self.__class__.__name__}(grid={self.grid}, axis={self.axis})"

    def __str__(self):
        return '"periodic"'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.grid == other.grid
            and self.axis == other.axis
        )

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ != other.__class__
            or self.grid != other.grid
            or self.axis != other.axis
        )

    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """
        return hash((self.grid._cache_hash(), self.axis))

    def copy(self) -> "BoundaryPeriodic":
        """ return a copy of itself, but with a reference to the same grid """
        return self.__class__(grid=self.grid, axis=self.axis)

    def extract_component(self, *indices):
        """extracts the boundary pair of the given extract_component.

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        return self

    def check_value_rank(self, rank: int):
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int): The rank of the value that is stored with this
                boundary condition
        """
        return True

    def make_virtual_point_evaluators(self) -> Tuple[Callable, Callable]:
        """returns two functions evaluating the value at virtual support points

        Returns:
            tuple: Two functions that each take a 1d array as an argument and
            return the associated value at the virtual support point outside the
            lower and upper boundary, respectively.
        """
        size = self.grid.shape[self.axis]

        @register_jitable
        def value_low(arr):
            """ evaluate the virtual point using the data array `arr` """
            return arr[size - 1]

        @register_jitable
        def value_high(arr):
            """ evaluate the virtual point using the data array `arr` """
            return arr[0]

        return (value_low, value_high)

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
        size = self.grid.shape[self.axis]
        if axis_coord == -1:
            # the virtual point on the lower side
            return 0, {size - 1: 1}
        elif axis_coord == size:
            # the virtual point on the upper side
            return 0, {0: 1}
        else:
            # the normal case of an interior point
            return 0, {axis_coord: 1}

    @property
    def differentiated(self) -> "BoundaryPeriodic":
        """ BoundaryPeriodic: differentiated boundary condition """
        return self

    def get_point_evaluator(self, fill: float = None) -> Callable:
        """return a function to evaluate values at a given point

        The point can either be a point inside the domain or a virtual point
        right outside the domain.

        Args:
            fill: This argument is ignored.

        Returns:
            function: A function taking a 1d array and an index as an argument,
                returning the value of the array at this index.
        """
        size = self.grid.shape[self.axis]

        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        @register_jitable
        def evaluate(arr, idx):
            """ evaluate values of the array `arr` at an index `idx` """
            arr_1d, i, _ = get_arr_1d(arr, idx)
            return arr_1d[..., i % size]  # wrap around for periodic boundaries

        return evaluate  # type: ignore

    def make_region_evaluator(self) -> Callable:
        """return a function to evaluate values in a neighborhood of a point

        Returns:
            function: A function that can be called with the data array and a
            tuple indicating around what point the region is evaluated. The
            function returns the data values left of the point, at the point,
            and right of the point along the axis associated with this boundary
            condition. The function takes boundary conditions into account if
            the point lies on the boundary.
        """
        size = self.grid.shape[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        @register_jitable
        def region_evaluator(arr, idx: Tuple[int, ...]) -> Tuple[float, float, float]:
            """ compiled function return the values in the region """
            # extract the 1d array along axis
            arr_1d, i, _ = get_arr_1d(arr, idx)

            # determine the indices in the vicinity
            im = size - 1 if i == 0 else i - 1
            ip = 0 if i == size - 1 else i + 1

            # return the values in the region around the point
            return arr_1d[..., im], arr_1d[..., i], arr_1d[..., ip]

        return region_evaluator  # type: ignore


def get_boundary_axis(
    grid: GridBase, axis: int, data, rank: int = 0
) -> BoundaryAxisBase:
    """return object representing the boundary condition for a single axis

    Args:
        grid (:class:`~pde.grids.GridBase`):
            The grid for which the boundary conditions are defined
        axis (int):
            The axis to which this boundary condition is associated
        data (str or tuple or dict):
            Data describing the boundary conditions for this axis
        rank (int):
            The tensorial rank of the value associated with the boundary
            conditions.

    Returns:
        BoundaryAxisBase: The boundary condition for the axis
    """
    # handle special constructs that describe boundary conditions
    if data == "natural" or data == "auto_periodic_neumann":
        # automatic choice between periodic and Neumann condition
        data = "periodic" if grid.periodic[axis] else "derivative"
    elif data == "auto_periodic_dirichlet":
        # automatic choice between periodic and Dirichlet condition
        data = "periodic" if grid.periodic[axis] else "value"

    # handle different types of data that specify boundary conditions
    if isinstance(data, BoundaryAxisBase):
        # boundary is already in the correct format
        return data
    elif data == "periodic" or data == ("periodic", "periodic"):
        # initialize a periodic boundary condition
        return BoundaryPeriodic(grid, axis)
    elif isinstance(data, dict) and data.get("type") == "periodic":
        # initialize a periodic boundary condition
        return BoundaryPeriodic(grid, axis)
    else:
        # initialize independent boundary conditions for the two sides
        return BoundaryPair.from_data(grid, axis, data, rank=rank)
