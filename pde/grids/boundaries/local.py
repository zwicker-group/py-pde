r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module contains classes for handling a single boundary of a non-periodic
axis. Since an axis has two boundary, we simply distinguish them by a boolean
flag `upper`, which is True for the side of the axis with the larger coordinate.

The module currently supports different boundary conditions:

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
  
There are also additional classes that impose boundary conditions only for the normal
components of fields, which can be important for vector and tensor fields. The classes
corresponding to the ones listed above are
:class:`~pde.grids.boundaries.local.DirichletNormalBC`,
:class:`~pde.grids.boundaries.local.NeumannNormalBC`,
:class:`~pde.grids.boundaries.local.MixedNormalBC`, and
:class:`~pde.grids.boundaries.local.CurvatureNormalBC`.

Note that derivatives are generally given in the direction of the outward normal vector,
such that positive derivatives correspond to a function that increases across the
boundary.
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numba as nb
import numpy as np
from numba.extending import register_jitable

from ...tools.docstrings import fill_in_docstring
from ...tools.numba import address_as_void_pointer
from ...tools.typing import (
    AdjacentEvaluator,
    FloatNumerical,
    GhostCellSetter,
    VirtualPointEvaluator,
)
from ..base import GridBase

BoundaryData = Union[Dict, str, "BCBase"]


class BCDataError(ValueError):
    """exception that signals that incompatible data was supplied for the BC"""

    pass


def _get_arr_1d(arr, idx: Tuple[int, ...], axis: int) -> Tuple[np.ndarray, int, Tuple]:
    """extract the 1d array along axis at point idx

    Args:
        arr (:class:`~numpy.ndarray`): The full data array
        idx (tuple): An index into the data array
        axis (int): The axis along which the 1d array will be extracted

    Returns:
        tuple: a tuple (arr_1d, i, bc_i), where `arr_1d` is the 1d array, `i` is
        the index `i` into this array marking the current point and `bc_i` are
        the remaining components of `idx`, which locate the point in the
        orthogonal directions. Consequently, `i = idx[axis]` and
        `arr[..., idx] == arr_1d[..., i]`.
    """
    dim = len(idx)
    # extract the correct indices
    if dim == 1:
        i = idx[0]
        bc_idx: Tuple = (...,)
        arr_1d = arr

    elif dim == 2:
        if axis == 0:
            i, y = idx
            bc_idx = (..., y)
            arr_1d = arr[..., :, y]
        elif axis == 1:
            x, i = idx
            bc_idx = (..., x)
            arr_1d = arr[..., x, :]

    elif dim == 3:
        if axis == 0:
            i, y, z = idx
            bc_idx = (..., y, z)
            arr_1d = arr[..., :, y, z]
        elif axis == 1:
            x, i, z = idx
            bc_idx = (..., x, z)
            arr_1d = arr[..., x, :, z]
        elif axis == 2:
            x, y, i = idx
            bc_idx = (..., x, y)
            arr_1d = arr[..., x, y, :]

    else:
        raise NotImplementedError

    return arr_1d, i, bc_idx


def _make_get_arr_1d(
    dim: int, axis: int
) -> Callable[[np.ndarray, Tuple[int, ...]], Tuple[np.ndarray, int, Tuple]]:
    """create function that extracts a 1d array at a given position

    Args:
        dim (int):
            The dimension of the space, i.e., the number of axes in the supplied array
        axis (int):
            The axis that is returned as the 1d array

    Returns:
        function: A numba compiled function that takes the full array `arr` and
        an index `idx` (a tuple of `dim` integers) specifying the point where
        the 1d array is extract. The function returns a tuple (arr_1d, i, bc_i),
        where `arr_1d` is the 1d array, `i` is the index `i` into this array
        marking the current point and `bc_i` are the remaining components of
        `idx`, which locate the point in the orthogonal directions.
        Consequently, `i = idx[axis]` and `arr[..., idx] == arr_1d[..., i]`.
    """
    assert 0 <= axis < dim
    ResultType = Tuple[np.ndarray, int, Tuple]

    # extract the correct indices
    if dim == 1:

        def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
            """extract the 1d array along axis at point idx"""
            i = idx[0]
            bc_idx: Tuple = (...,)
            arr_1d = arr
            return arr_1d, i, bc_idx

    elif dim == 2:
        if axis == 0:

            def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                i, y = idx
                bc_idx = (..., y)
                arr_1d = arr[..., :, y]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, i = idx
                bc_idx = (..., x)
                arr_1d = arr[..., x, :]
                return arr_1d, i, bc_idx

    elif dim == 3:
        if axis == 0:

            def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                i, y, z = idx
                bc_idx = (..., y, z)
                arr_1d = arr[..., :, y, z]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, i, z = idx
                bc_idx = (..., x, z)
                arr_1d = arr[..., x, :, z]
                return arr_1d, i, bc_idx

        elif axis == 2:

            def get_arr_1d(arr: np.ndarray, idx: Tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, y, i = idx
                bc_idx = (..., x, y)
                arr_1d = arr[..., x, y, :]
                return arr_1d, i, bc_idx

    else:
        raise NotImplementedError

    return register_jitable(inline="always")(get_arr_1d)  # type: ignore


class BCBase(metaclass=ABCMeta):
    """represents a single boundary in an BoundaryPair instance"""

    names: List[str]
    """ list: identifiers used to specify the given boundary class """
    normal: bool = False
    """ bool: Flag indicating whether only the normal components are affected"""
    homogeneous: bool
    """ bool: determines whether the boundary condition depends on space """

    _subclasses: Dict[str, Type[BCBase]] = {}  # all classes inheriting from this
    _conditions: Dict[str, Type[BCBase]] = {}  # mapping from all names to classes

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
    ):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
        """
        self.grid = grid
        self.axis = axis
        self.upper = upper
        self.rank = rank

        # Get the shape of the values imposed at the boundary. These are the shape of
        # the tensor field unless only the normal component is specified
        if self.rank == 0:
            self.normal = False
        if self.normal:
            self._shape_tensor = (self.grid.dim,) * (self.rank - 1)
        else:
            self._shape_tensor = (self.grid.dim,) * self.rank

        # get the shape of the data at the boundary
        self._shape_boundary = (
            self.grid.shape[: self.axis] + self.grid.shape[self.axis + 1 :]
        )

        self._logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
        if hasattr(cls, "names"):
            for name in cls.names:
                cls._conditions[name] = cls

    @property
    def periodic(self) -> bool:
        """bool: whether the axis is periodic"""
        return self.grid.periodic[self.axis]

    @property
    def axis_coord(self) -> float:
        """float: value of the coordinate that defines this boundary condition"""
        if self.upper:
            return self.grid.axes_bounds[self.axis][1]
        else:
            return self.grid.axes_bounds[self.axis][0]

    @classmethod
    def get_help(cls) -> str:
        """Return information on how boundary conditions can be set"""
        types = ", ".join(
            f"'{subclass.names[0]}'"
            for subclass in cls._subclasses.values()
            if hasattr(subclass, "names")
        )
        return (
            f"Possible types of boundary conditions are {types}. "
            "Values can be set using {'type': TYPE, 'value': VALUE}. "
            "Here, VALUE can be a scalar number, a vector for tensorial boundary "
            "conditions, or a string, which can be interpreted as a sympy expression. "
            "In the latter case, the names of the axes not associated with this "
            "boundary can be used as variables to describe inhomogeneous boundary "
            "conditions."
        )

    @abstractmethod
    def _repr_value(self) -> List[str]:
        pass

    def __repr__(self):
        args = [f"axis={self.axis}", f"upper={self.upper}"]
        if self.rank != 1:
            args.append(f"rank={self.rank}")
        args += self._repr_value()
        return f"{self.__class__.__name__}({', '.join(args)})"

    __str__ = __repr__

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.grid == other.grid
            and self.axis == other.axis
            and self.homogeneous == other.homogeneous
            and self.rank == other.rank
        )

    def __ne__(self, other):
        return not self == other

    @classmethod
    def from_str(
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        condition: str,
        rank: int = 0,
        **kwargs,
    ) -> BCBase:
        r"""creates boundary from a given string identifier

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            condition (str):
                Identifies the boundary condition
            rank (int):
                The tensorial rank of the field for this boundary condition
            \**kwargs:
                Additional arguments passed to the constructor
        """
        # raise warning to mention problem with legacy code (bug fixed 2021-01-18)
        if condition == "no-flux":
            raise BCDataError(
                "Specifying the boundary condition 'no-flux' is no longer supported "
                "since it introduced a bug when specifying flux conditions. To impose "
                "no flux conditions, please decide whether you need to impose a "
                "vanishing derivative or a value of zero and specify this condition "
                "explicitly."
            )

        # extract the class
        try:
            boundary_class = cls._conditions[condition]
        except KeyError:
            raise BCDataError(
                f"Boundary condition `{condition}` not defined. " + cls.get_help()
            )

        # create the actual class
        return boundary_class(grid=grid, axis=axis, upper=upper, rank=rank, **kwargs)

    @classmethod
    def from_dict(
        cls, grid: GridBase, axis: int, upper: bool, data: Dict[str, Any], rank: int = 0
    ) -> BCBase:
        """create boundary from data given in dictionary

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (dict):
                The dictionary defining the boundary condition
            rank (int):
                The tensorial rank of the field for this boundary condition
        """
        data = data.copy()  # need to make a copy since we modify it below

        # parse all possible variants that could be given
        if "type" in data.keys():
            # type is given (optionally with a value)
            b_type = data.pop("type")
            b_value = data.pop("value", 0)

        elif len(data) == 1:
            # only a single items is given
            b_type, b_value = data.popitem()

        else:
            raise BCDataError(
                f"Boundary conditions `{str(list(data.keys()))}` are not supported."
            )

        # initialize the boundary class with all remaining values forwarded
        return cls.from_str(
            grid, axis, upper, condition=b_type, rank=rank, value=b_value, **data
        )

    @classmethod
    def from_data(
        cls, grid: GridBase, axis: int, upper: bool, data: BoundaryData, rank: int = 0
    ) -> BCBase:
        """create boundary from some data

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (str or dict):
                Data that describes the boundary
            rank (int):
                The tensorial rank of the field for this boundary condition

        Returns:
            :class:`~pde.grids.boundaries.local.BCBase`: the instance created
            from the data

        Throws:
            ValueError if `data` cannot be interpreted as a boundary condition
        """
        # check all different data formats
        if isinstance(data, BCBase):
            # already in the correct format
            assert data.grid == grid and data.axis == axis and data.rank == rank
            return data.copy(upper=upper)

        elif isinstance(data, dict):
            # create from dictionary
            return cls.from_dict(grid, axis, upper=upper, data=data, rank=rank)

        elif isinstance(data, str):
            # create a specific condition given by a string
            return cls.from_str(grid, axis, upper=upper, condition=data, rank=rank)

        else:
            raise BCDataError(
                f"Unsupported boundary format: `{data}`. " + cls.get_help()
            )

    def check_value_rank(self, rank: int) -> None:
        """check whether the values at the boundaries have the correct rank

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition

        Throws:
            RuntimeError: if the value does not have rank `rank`
        """
        if self.rank != rank:
            raise RuntimeError(
                f"Expected rank {rank}, but boundary condition had rank {self.rank}."
            )

    @abstractmethod
    def _cache_hash(self) -> int:
        pass

    @abstractmethod
    def copy(self, upper: Optional[bool] = None, rank: int = None) -> BCBase:
        pass

    @abstractmethod
    def extract_component(self, *indices):
        pass

    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        pass

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        raise NotImplementedError

    @abstractmethod
    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for this boundary"""

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells for this boundary"""
        normal = self.normal
        axis = self.axis

        # get information of the virtual points (ghost cells)
        vp_idx = self.grid.shape[self.axis] + 1 if self.upper else 0
        np_idx = self.grid.shape[self.axis] - 1 if self.upper else 0
        vp_value = self.make_virtual_point_evaluator()

        if self.grid.num_axes == 1:  # 1d grid

            @register_jitable
            def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                """helper function setting the conditions on all axes"""
                data_valid = data_full[..., 1:-1]
                val = vp_value(data_valid, (np_idx,), args=args)
                if normal:
                    data_full[..., axis, vp_idx] = val
                else:
                    data_full[..., vp_idx] = val

        elif self.grid.num_axes == 2:  # 2d grid

            if self.axis == 0:
                num_y = self.grid.shape[1]

                @register_jitable
                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    """helper function setting the conditions on all axes"""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for j in range(num_y):
                        val = vp_value(data_valid, (np_idx, j), args=args)
                        if normal:
                            data_full[..., axis, vp_idx, j + 1] = val
                        else:
                            data_full[..., vp_idx, j + 1] = val

            elif self.axis == 1:
                num_x = self.grid.shape[0]

                @register_jitable
                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    """helper function setting the conditions on all axes"""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for i in range(num_x):
                        val = vp_value(data_valid, (i, np_idx), args=args)
                        if normal:
                            data_full[..., axis, i + 1, vp_idx] = val
                        else:
                            data_full[..., i + 1, vp_idx] = val

        elif self.grid.num_axes == 3:  # 3d grid

            if self.axis == 0:
                num_y, num_z = self.grid.shape[1:]

                @register_jitable
                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    """helper function setting the conditions on all axes"""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for j in range(num_y):
                        for k in range(num_z):
                            val = vp_value(data_valid, (np_idx, j, k), args=args)
                            if normal:
                                data_full[..., axis, vp_idx, j + 1, k + 1] = val
                            else:
                                data_full[..., vp_idx, j + 1, k + 1] = val

            elif self.axis == 1:
                num_x, num_z = self.grid.shape[0], self.grid.shape[2]

                @register_jitable
                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    """helper function setting the conditions on all axes"""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for k in range(num_z):
                            val = vp_value(data_valid, (i, np_idx, k), args=args)
                            if normal:
                                data_full[..., axis, i + 1, vp_idx, k + 1] = val
                            else:
                                data_full[..., i + 1, vp_idx, k + 1] = val

            elif self.axis == 2:
                num_x, num_y = self.grid.shape[:2]

                @register_jitable
                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    """helper function setting the conditions on all axes"""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for j in range(num_y):
                            val = vp_value(data_valid, (i, j, np_idx), args=args)
                            if normal:
                                data_full[..., axis, i + 1, j + 1, vp_idx] = val
                            else:
                                data_full[..., i + 1, j + 1, vp_idx] = val

        else:
            raise NotImplementedError("Too many axes")

        return ghost_cell_setter  # type: ignore


class ExpressionBC(BCBase):
    """represents a boundary whose virtual point is calculated from an expression"""

    names = ["virtual_point"]

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: Union[float, str] = 0,
        target: str = "virtual_point",
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str):
                An expression that determines the value of the boundary condition.
            target (str):
                Selects which value is actually set. Possible choices include `value`,
                `derivative`, and `virtual_point`.
        """
        super().__init__(grid, axis, upper, rank=rank)

        if self.rank != 0:
            raise NotImplementedError(
                "Expression boundary conditions only work for scalar conditions"
            )
        self._logger = logging.getLogger(self.__class__.__name__)

        # determine the full expression for setting the value of the virtual point
        if target == "virtual_point":
            expression = value
        elif target == "value":
            expression = f"2 * ({value}) - value"
        elif target == "derivative":
            expression = f"dx * ({value}) + value"
        else:
            raise ValueError(f"Unknown target `{target}` for expression")

        # parse this expression
        from pde.tools.expressions import ScalarExpression

        signature = ["value", "dx"] + grid.axes + ["t"]
        self._expr = ScalarExpression(expression, signature=signature)

        # quickly check whether the expression was parsed correctly
        test_value = np.zeros((self.grid.dim,) * self.rank)
        dx = self.grid.discretization[self.axis]
        coords = tuple(bounds[0] for bounds in grid.axes_bounds)
        try:
            self._expr(test_value, dx, *coords, t=0)
        except Exception as err:
            raise BCDataError(
                f"Could not evaluate BC expression `{expression}` with signature "
                f"{signature}.\nEncountered error: {err}"
            )

    def _repr_value(self):
        return [f'value="{self._expr.expression}"']

    def _cache_hash(self) -> int:
        """returns a value to determine when a cache needs to be updated"""
        expression = self._expr.expression
        return hash(
            (self.__class__.__name__, self.grid._cache_hash(), self.axis, expression)
        )

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self._expr.expression == other._expr.expression

    def copy(
        self,
        upper: Optional[bool] = None,
        rank: int = None,
    ) -> ExpressionBC:
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self._expr.expression,
        )

    def extract_component(self, *indices):
        """extracts the boundary conditions for the given component

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        raise NotImplementedError

    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        raise NotImplementedError

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        raise NotImplementedError

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for this boundary

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        dx = self.grid.discretization[self.axis]

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: List[Union[slice, int]] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_read = idx_write[:]
        idx_read[offset + self.axis] = -2 if self.upper else 1

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_read[offset - 1] = self.axis

        # prepare the arguments
        values = data_full[tuple(idx_read)]
        coords = self.grid._boundary_coordinates(axis=self.axis, upper=self.upper)
        coords = np.moveaxis(coords, -1, 0)  # point coordinates to first axis

        if args is None:
            if self._expr.depends_on("t"):
                raise RuntimeError(
                    "Require value for `t` for time-dependent BC. The value must be "
                    "passed explicitly via `args` when calling a differential operator."
                )
            t = 0.0
        else:
            t = float(args["t"])

        # calculate the virtual points
        data_full[tuple(idx_write)] = self._expr(values, dx, *coords, t)

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        """returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """
        dx = self.grid.discretization[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        time_dependent = self._expr.depends_on("t")
        bc_coords = self.grid._boundary_coordinates(axis=self.axis, upper=self.upper)
        bc_coords = np.moveaxis(bc_coords, -1, 0)  # point coordinates to first axis
        func = self._expr.get_compiled()
        dim = self.grid.dim
        assert dim <= 3

        @register_jitable
        def virtual_point(arr: np.ndarray, idx: Tuple[int, ...], args=None) -> float:
            """evaluate the virtual point at `idx`"""
            _, _, bc_idx = get_arr_1d(arr, idx)
            grid_value = arr[idx]
            coords = bc_coords[bc_idx]

            # extract time for handling time-dependent BCs
            if args is None:
                if time_dependent:
                    raise RuntimeError(
                        "Require value for `t` for time-dependent BC. The value must "
                        "be passed explicitly via `args` when calling a differential "
                        "operator."
                    )
                t = 0.0
            else:
                t = float(args["t"])

            if dim == 1:
                return func(grid_value, dx, coords[0], t)  # type: ignore
            elif dim == 2:
                return func(grid_value, dx, coords[0], coords[1], t)  # type: ignore
            elif dim == 3:
                return func(grid_value, dx, coords[0], coords[1], coords[2], t)  # type: ignore
            else:
                return np.nan  #  cheap way to signal a problem

        return virtual_point  # type: ignore


class ExpressionValueBC(ExpressionBC):
    """represents a boundary whose value is calculated from an expression"""

    names = ["value_expression", "value_expr"]

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: Union[float, str] = 0,
        target: str = "value",
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str):
                An expression that determines the value of the boundary condition.
            target (str):
                Selects which value is actually set. Possible choices include `value`,
                `derivative`, and `virtual_point`.
        """
        super().__init__(grid, axis, upper, rank=rank, value=value, target=target)


class ExpressionDerivativeBC(ExpressionBC):
    """represents a boundary whose outward derivative is calculated from an expression"""

    names = ["derivative_expression", "derivative_expr"]

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: Union[float, str] = 0,
        target: str = "derivative",
    ):
        """
        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str):
                An expression that determines the value of the boundary condition.
            target (str):
                Selects which value is actually set. Possible choices include `value`,
                `derivative`, and `virtual_point`.
        """
        super().__init__(grid, axis, upper, rank=rank, value=value, target=target)


class ConstBCBase(BCBase):
    """base class representing a boundary whose virtual point is set from constants"""

    _value: np.ndarray

    value_is_linked: bool
    """ bool: flag that indicates whether the value associated with this
    boundary condition is linked to :class:`~numpy.ndarray` managed by external
    code. """

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: Union[float, np.ndarray, str] = 0,
    ):
        """
        Warning:
            {WARNING_EXEC} However, the function is safe when `value` cannot be
            an arbitrary string.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str or :class:`~numpy.ndarray`):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition. If
                value is a single value (or tensor in case of tensorial boundary
                conditions), the same value is applied to all points.
                Inhomogeneous boundary conditions are possible by supplying an
                expression as a string, which then may depend on the axes names
                of the respective grid.
        """
        super().__init__(grid, axis, upper, rank=rank)
        self.value = value  # type: ignore

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and np.array_equal(self.value, other.value)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter  # type: ignore
    @fill_in_docstring
    def value(self, value: Union[float, np.ndarray, str] = 0):
        """set the value of this boundary condition

        Warning:
            {WARNING_EXEC}

        Args:
            value (float or str or array):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition.
        """
        self._value = self._parse_value(value)

        if self._value.shape == self._shape_tensor:
            # value does not depend on space
            self.homogeneous = True
        elif self._value.shape == self._shape_tensor + self._shape_boundary:
            # inhomogeneous field
            self.homogeneous = False
        else:
            raise ValueError(
                f"Dimensions {self._value.shape} of the value are incompatible with "
                f"rank {self.rank} and spatial dimensions {self._shape_boundary}"
            )

        self.value_is_linked = False

    def _repr_value(self):
        if self.value_is_linked:
            return [f"value=<linked: {self.value.ctypes.data}>"]
        elif np.array_equal(self.value, 0):
            return []
        else:
            return [f"value={self.value!r}"]

    def __str__(self):
        if hasattr(self, "names"):
            if np.array_equal(self.value, 0):
                return f'"{self.names[0]}"'
            elif self.value_is_linked:
                return (
                    f'{{"type": "{self.names[0]}", '
                    f'"value": <linked: {self.value.ctypes.data}>}}'
                )
            else:
                return f'{{"type": "{self.names[0]}", "value": {self.value}}}'
        else:
            return self.__repr__()

    @fill_in_docstring
    def _parse_value(self, value: Union[float, np.ndarray, str]) -> np.ndarray:
        """parses a boundary value

        Warning:
            {WARNING_EXEC}

        Args:
            value (array-like or str):
                The value given as a array of tensorial character and optionally
                dependent on space (along the boundary). Alternatively, a string
                can specify a mathematical expression that can optionally depend
                on the coordinates along the boundary. This expression is only
                supported for scalar boundary conditions.

        Returns:
            :class:`~numpy.ndarray`: The value at the boundary
        """
        if isinstance(value, str):
            # inhomogeneous value given by an expression
            if len(self._shape_tensor) > 0:
                raise NotImplementedError(
                    "Expressions for boundaries are only supported for scalar values."
                )

            from ...tools.expressions import ScalarExpression

            # determine which coordinates are allowed to vary
            axes_ids = list(range(self.axis)) + list(
                range(self.axis + 1, self.grid.num_axes)
            )

            # parse the expression with the correct variables
            bc_vars = [self.grid.axes[i] for i in axes_ids]
            expr = ScalarExpression(value, self.grid.axes)

            if axes_ids:
                # extended boundary

                # get the coordinates at each point of the boundary
                bc_coords = np.meshgrid(
                    *[self.grid.axes_coords[i] for i in axes_ids], indexing="ij"
                )

                # determine the value at each of these points. Note that we here
                # iterate explicitly over all points because the expression might
                # not depend on some of the variables, but we still want the array
                # to contain a value at each boundary point
                result = np.empty_like(bc_coords[0])
                coords: Dict[str, float] = {name: 0 for name in self.grid.axes}
                # set the coordinate of this BC
                coords[self.grid.axes[self.axis]] = self.axis_coord
                for idx in np.ndindex(*result.shape):
                    for i, name in enumerate(bc_vars):
                        coords[name] = bc_coords[i][idx]
                    result[idx] = expr(**coords)

            else:
                # point boundary
                result = np.array(expr(self.axis_coord))

        elif np.isscalar(value):
            # scalar value applied to all positions
            result = np.broadcast_to(float(value), self._shape_tensor)

        else:
            # assume tensorial and/or inhomogeneous values
            value = np.asarray(value, dtype=np.double)

            if value.ndim == 0:
                # value is a scalar
                result = np.broadcast_to(value, self._shape_tensor)
            elif value.shape == self._shape_tensor + self._shape_boundary:
                # inhomogeneous field with all tensor components
                result = value
            elif value.shape == self._shape_tensor:
                # homogeneous field with all tensor components
                result = value
            else:
                raise ValueError(
                    f"Dimensions {value.shape} of the given value are incompatible "
                    f"with the expected shape {self._shape_tensor} of the boundary "
                    f"value and its spatial dimensions {self._shape_boundary}."
                )

        # check consistency
        if np.any(np.isnan(result)):
            try:
                logger = self._logger
            except AttributeError:
                # this can happen when _parse_value is called before the object
                # is fully initialized
                logger = logging.getLogger(self.__class__.__name__)
            logger.warning("In valid values in %s", self)

        return result  # type: ignore

    def link_value(self, value: np.ndarray):
        """link value of this boundary condition to external array"""
        assert value.data.c_contiguous

        shape = self._shape_tensor + self._shape_boundary
        if value.shape != shape:
            raise ValueError(
                f"The shape of the value, {value.shape}, is incompatible with the "
                f"expected shape for this boundary condition, {shape}"
            )
        self._value = value
        self.homogeneous = False
        self.value_is_linked = True

    def _cache_hash(self) -> int:
        """returns a value to determine when a cache needs to be updated"""
        if self.value_is_linked:
            value: Union[int, bytes] = self.value.ctypes.data
        else:
            value = self.value.tobytes()

        return hash(
            (self.__class__.__name__, self.grid._cache_hash(), self.axis, value)
        )

    def copy(
        self,
        upper: Optional[bool] = None,
        rank: int = None,
        value: Union[float, np.ndarray, str] = None,
    ) -> ConstBCBase:
        """return a copy of itself, but with a reference to the same grid"""
        obj = self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self.value if value is None else value,
        )
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj

    def extract_component(self, *indices):
        """extracts the boundary conditions for the given component

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        return self.copy(value=self.value[indices], rank=self.rank - len(indices))

    def _make_value_getter(self) -> Callable[[], np.ndarray]:
        """return a (compiled) function for obtaining the value.

        Note:
            This should only be used in numba compiled functions that need to
            support boundary values that can be changed after the function has
            been compiled. In essence, the helper function created here serves
            to get around the compile-time constants that are otherwise created.

        Warning:
            The returned function has a hard-coded reference to the memory
            address of the value error, which must thus be maintained in memory.
            If the address of self.value changes, a new function needs to be
            created by calling this factory function again.
        """
        # obtain details about the array
        mem_addr = self.value.ctypes.data
        shape = self.value.shape
        dtype = self.value.dtype

        # Note that we tried using register_jitable here, but this lead to
        # problems with address_as_void_pointer

        @nb.jit(nb.typeof(self._value)(), inline="always")
        def get_value() -> np.ndarray:
            """helper function returning the linked array"""
            return nb.carray(address_as_void_pointer(mem_addr), shape, dtype)  # type: ignore

        # keep a reference to the array to prevent garbage collection
        get_value._value_ref = self._value

        return get_value  # type: ignore


class ConstBC1stOrderBase(ConstBCBase):
    """represents a single boundary in an BoundaryPair instance"""

    @abstractmethod
    def get_virtual_point_data(self, compiled: bool = False) -> Tuple[Any, Any, int]:
        pass

    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        """sets the elements of the sparse representation of this condition

        Args:
            idx (tuple):
                The index of the point that must lie on the boundary condition

        Returns:
            float, dict: A constant value and a dictionary with indices and
            factors that can be used to calculate this virtual point
        """
        data = self.get_virtual_point_data()

        if self.homogeneous:
            const = data[0]
            factor = data[1]
        else:
            # obtain index of the boundary point
            idx_c = list(idx)
            del idx_c[self.axis]
            const = data[0][tuple(idx_c)]
            factor = data[1][tuple(idx_c)]

        return const, {data[2]: factor}

    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        """calculate the value of the virtual point outside the boundary

        Args:
            arr (array):
                The data values associated with the grid
            idx (tuple):
                The index of the point to evaluate. This is a tuple of length
                `grid.num_axes` with the either -1 or `dim` as the entry for the
                axis associated with this boundary condition. Here, `dim` is the
                dimension of the axis. The index is optional if dim == 1.

        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                raise ValueError(
                    "Index `idx` can only be deduced for grids with a single axis."
                )

        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)

        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()

        if self.homogeneous:
            return const + factor * arr_1d[..., index]  # type: ignore
        else:
            return const[bc_idx] + factor[bc_idx] * arr_1d[..., index]  # type: ignore

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        """returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """
        normal = self.normal
        axis = self.axis
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data(compiled=True)

        if self.homogeneous:

            @nb.jit
            def virtual_point(
                arr: np.ndarray, idx: Tuple[int, ...], args=None
            ) -> float:
                """evaluate the virtual point at `idx`"""
                arr_1d, _, _ = get_arr_1d(arr, idx)
                if normal:
                    val_field = arr_1d[..., axis, index]
                else:
                    val_field = arr_1d[..., index]
                return const() + factor() * val_field  # type: ignore

        else:

            @nb.jit
            def virtual_point(
                arr: np.ndarray, idx: Tuple[int, ...], args=None
            ) -> float:
                """evaluate the virtual point at `idx`"""
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                if normal:
                    val_field = arr_1d[..., axis, index]
                else:
                    val_field = arr_1d[..., index]
                return const()[bc_idx] + factor()[bc_idx] * val_field  # type: ignore

        return virtual_point  # type: ignore

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        """returns a function evaluating the value adjacent to a given point

        Returns:
            function: A function with signature (arr_1d, i_point, bc_idx), where
            `arr_1d` is the one-dimensional data array (the data points along
            the axis perpendicular to the boundary), `i_point` is the index into
            this array for the current point and bc_idx are the remaining
            indices of the current point, which indicate the location on the
            boundary plane. The result of the function is the data value at the
            adjacent point along the axis associated with this boundary
            condition in the upper (lower) direction when `upper` is True
            (False).
        """
        # get values distinguishing upper from lower boundary
        if self.upper:
            i_bndry = self.grid.shape[self.axis] - 1
            i_dx = 1
        else:
            i_bndry = 0
            i_dx = -1

        if self.homogeneous:
            # the boundary condition does not depend on space

            # calculate necessary constants
            const, factor, index = self.get_virtual_point_data(compiled=True)
            zeros = np.zeros(self._shape_tensor)
            ones = np.ones(self._shape_tensor)

            @register_jitable(inline="always")
            def adjacent_point(
                arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
            ) -> FloatNumerical:
                """evaluate the value adjacent to the current point"""
                # determine the parameters for evaluating adjacent point. Note
                # that defining the variables c and f for the interior points
                # seems needless, but it turns out that this results in a 10x
                # faster function (because of branch prediction?).
                if i_point == i_bndry:
                    c, f, i = const(), factor(), index
                else:
                    c, f, i = zeros, ones, i_point + i_dx  # INTENTIONAL

                # calculate the values
                return c + f * arr_1d[..., i]  # type: ignore

        else:
            # the boundary condition is a function of space

            # calculate necessary constants
            const, factor, index = self.get_virtual_point_data(compiled=True)
            zeros = np.zeros(self._shape_tensor + self._shape_boundary)
            ones = np.ones(self._shape_tensor + self._shape_boundary)

            @register_jitable(inline="always")
            def adjacent_point(arr_1d, i_point, bc_idx) -> float:
                """evaluate the value adjacent to the current point"""
                # determine the parameters for evaluating adjacent point. Note
                # that defining the variables c and f for the interior points
                # seems needless, but it turns out that this results in a 10x
                # faster function (because of branch prediction?). This is
                # surprising, because it uses arrays zeros and ones that are
                # quite pointless
                if i_point == i_bndry:
                    c, f, i = const(), factor(), index
                else:
                    c, f, i = zeros, ones, i_point + i_dx  # INTENTIONAL

                # calculate the values
                return c[bc_idx] + f[bc_idx] * arr_1d[..., i]  # type: ignore

        return adjacent_point  # type: ignore

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for this boundary

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: List[Union[slice, int]] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_read = idx_write[:]
        idx_read[offset + self.axis] = index + 1

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_read[offset - 1] = self.axis

        if self.homogeneous and not np.isscalar(factor):
            # add dimension to const so it can be broadcasted to shape of data_full
            for _ in range(self.grid.num_axes - 1):
                factor = factor[..., np.newaxis]
                const = const[..., np.newaxis]

        # calculate the virtual points
        data_full[tuple(idx_write)] = const + factor * data_full[tuple(idx_read)]


class _PeriodicBC(ConstBC1stOrderBase):
    """represents one part of a boundary condition"""

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        flip_sign: bool = False,
    ):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            flip_sign (bool):
                Impose different signs on the two sides of the boundary
        """
        super().__init__(grid, axis, upper)
        self.flip_sign = flip_sign

    def get_virtual_point_data(self, compiled: bool = False) -> Tuple[Any, Any, int]:
        """return data suitable for calculating virtual points

        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.

        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """
        index = 0 if self.upper else self.grid.shape[self.axis] - 1
        value = -1 if self.flip_sign else 1

        if not compiled:
            return (0.0, value, index)
        else:
            const = np.array(0, np.double)
            factor = np.array(value, np.double)

            @register_jitable(inline="always")
            def const_func():
                return const

            @register_jitable(inline="always")
            def factor_func():
                return factor

            return (const_func, factor_func, index)


class DirichletBC(ConstBC1stOrderBase):
    """represents a boundary condition imposing the value"""

    names = ["value", "dirichlet"]  # identifiers for this boundary condition
    normal = False

    def get_virtual_point_data(self, compiled: bool = False) -> Tuple[Any, Any, int]:
        """return data suitable for calculating virtual points

        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.

        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """
        const = 2 * self.value
        index = self.grid.shape[self.axis] - 1 if self.upper else 0

        if not compiled:
            factor = -np.ones_like(const)
            return (const, factor, index)
        else:
            # return boundary data such that dynamically calculated values can
            # be used in numba compiled code. This is a work-around since numpy
            # arrays are copied into closures, making them compile-time
            # constants

            const = np.array(const, np.double)
            factor = np.full_like(const, -1)

            if self.value_is_linked:
                value = self._make_value_getter()

                @register_jitable(inline="always")
                def const_func():
                    return 2 * value()

            else:

                @register_jitable(inline="always")
                def const_func():
                    return const

            @register_jitable(inline="always")
            def factor_func():
                return factor

            return (const_func, factor_func, index)


class DirichletNormalBC(DirichletBC):
    """represents a boundary condition imposing the normal component of a value"""

    names = ["normal_value", "normal_component", "value_normal", "dirichlet_normal"]
    normal = True


class NeumannBC(ConstBC1stOrderBase):
    """represents a boundary condition imposing the derivative in the outward
    normal direction of the boundary"""

    names = ["derivative", "neumann"]  # identifiers for this boundary condition
    normal = False

    def get_virtual_point_data(self, compiled: bool = False) -> Tuple[Any, Any, int]:
        """return data suitable for calculating virtual points

        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.

        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """
        dx = self.grid.discretization[self.axis]

        const = dx * self.value
        index = self.grid.shape[self.axis] - 1 if self.upper else 0

        if not compiled:
            factor = np.ones_like(const)
            return (const, factor, index)
        else:
            # return boundary data such that dynamically calculated values can
            # be used in numba compiled code. This is a work-around since numpy
            # arrays are copied into closures, making them compile-time
            # constants

            const = np.array(const, np.double)
            factor = np.ones_like(const)

            if self.value_is_linked:
                value = self._make_value_getter()

                @register_jitable(inline="always")
                def const_func():
                    return dx * value()

            else:

                @register_jitable(inline="always")
                def const_func():
                    return const

            @register_jitable(inline="always")
            def factor_func():
                return factor

            return (const_func, factor_func, index)


class NeumannNormalBC(NeumannBC):
    """represents a boundary condition imposing the derivative in the outward
    normal direction of the boundary only on the normal component of the tensor field"""

    names = ["derivative_normal", "neumann_normal"]
    normal = True


class MixedBC(ConstBC1stOrderBase):
    r"""represents a mixed (or Robin) boundary condition imposing a derivative
    in the outward normal direction of the boundary that is given by an affine
    function involving the actual value:

    .. math::
        \partial_n c + \gamma c = \beta

    Here, :math:`c` is the field to which the condition is applied,
    :math:`\gamma` quantifies the influence of the field and :math:`\beta` is
    the constant term. Note that :math:`\gamma = 0` corresponds
    to Dirichlet conditions imposing :math:`\beta` as the derivative.
    Conversely,  :math:`\gamma \rightarrow \infty` corresponds to imposing a
    zero value on :math:`c`.

    This condition can be enforced by using one of the following variants

    .. code-block:: python

        bc = {'mixed': VALUE}
        bc = {'type': 'mixed', 'value': VALUE, 'const': CONST}

    where `VALUE` corresponds to :math:`\gamma` and `CONST` to :math:`\beta`.
    """

    names = ["mixed", "robin"]
    normal = False

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: Union[float, np.ndarray, str] = 0,
        const: Union[float, np.ndarray, str] = 0,
    ):
        r"""
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated
                with the upper side of an axis or not. In essence, this
                determines the direction of the local normal vector of the
                boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str or array):
                The parameter :math:`\gamma` quantifying the influence of the
                field onto its normal derivative. If `value` is a single value
                (or tensor in case of tensorial boundary conditions), the same
                value is applied to all points.  Inhomogeneous boundary
                conditions are possible by supplying an expression as a string,
                which then may depend on the axes names of the respective grid.
            const (float or :class:`~numpy.ndarray` or str):
                The parameter :math:`\beta` determining the constant term for
                the boundary condition. Supports the same input as `value`.
        """
        super().__init__(grid, axis, upper, rank=rank, value=value)
        self.const = self._parse_value(const)

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.const == other.const

    def _cache_hash(self) -> int:
        """returns a value to determine when a cache needs to be updated"""
        return hash((super()._cache_hash(), self.const.tobytes()))

    def copy(
        self,
        upper: Optional[bool] = None,
        rank: int = None,
        value: Union[float, np.ndarray, str] = None,
        const: Union[float, np.ndarray, str] = None,
    ) -> "MixedBC":
        """return a copy of itself, but with a reference to the same grid"""
        obj = self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self.value if value is None else value,
            const=self.const if const is None else const,
        )
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj

    def get_virtual_point_data(self, compiled: bool = False) -> Tuple[Any, Any, int]:
        """return data suitable for calculating virtual points

        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.

        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """
        # calculate values assuming finite factor
        dx = self.grid.discretization[self.axis]
        with np.errstate(invalid="ignore"):
            const = np.asarray(2 * dx * self.const / (2 + dx * self.value))
            factor = np.asarray((2 - dx * self.value) / (2 + dx * self.value))

        # correct at places of infinite values
        const[~np.isfinite(factor)] = 0
        factor[~np.isfinite(factor)] = -1

        index = self.grid.shape[self.axis] - 1 if self.upper else 0

        if not compiled:
            return (const, factor, index)

        # return boundary data such that dynamically calculated values can
        # be used in numba compiled code. This is a work-around since numpy
        # arrays are copied into closures, making them compile-time
        # constants
        if self.value_is_linked:
            const_val = np.array(self.const, np.double)
            value_func = self._make_value_getter()

            @register_jitable(inline="always")
            def const_func():
                value = value_func()
                const = np.empty_like(value)
                for i in range(value.size):
                    val = value.flat[i]
                    if np.isinf(val):
                        const.flat[i] = 0
                    else:
                        const.flat[i] = 2 * dx * const_val / (2 + dx * val)
                return const

            @register_jitable(inline="always")
            def factor_func():
                value = value_func()
                factor = np.empty_like(value)
                for i in range(value.size):
                    val = value.flat[i]
                    if np.isinf(val):
                        factor.flat[i] = -1
                    else:
                        factor.flat[i] = (2 - dx * val) / (2 + dx * val)
                return factor

        else:
            const = np.array(const, np.double)
            factor = np.array(factor, np.double)

            @register_jitable(inline="always")
            def const_func():
                return const

            @register_jitable(inline="always")
            def factor_func():
                return factor

        return (const_func, factor_func, index)


class MixedNormalBC(MixedBC):
    r"""represents a mixed (or Robin) boundary condition imposing a derivative
    in the outward normal direction of the boundary only for the normal component"""

    names = ["mixed_normal", "robin_normal"]
    normal = True


class ConstBC2ndOrderBase(ConstBCBase):
    """abstract base class for boundary conditions of 2nd order"""

    @abstractmethod
    def get_virtual_point_data(self) -> Tuple[Any, Any, int, Any, int]:
        """return data suitable for calculating virtual points

        Returns:
            tuple: the data associated with this virtual point
        """

    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        """sets the elements of the sparse representation of this condition

        Args:
            idx (tuple):
                The index of the point that must lie on the boundary condition

        Returns:
            float, dict: A constant value and a dictionary with indices and
            factors that can be used to calculate this virtual point
        """
        data = self.get_virtual_point_data()

        if self.homogeneous:
            const = data[0]
            factor1 = data[1]
            factor2 = data[3]
        else:
            # obtain index of the boundary point
            idx_c = list(idx)
            del idx_c[self.axis]
            bc_idx = tuple(idx_c)
            const = data[0][bc_idx]
            factor1 = data[1][bc_idx]
            factor2 = data[3][bc_idx]

        return const, {data[2]: factor1, data[4]: factor2}

    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        """calculate the value of the virtual point outside the boundary

        Args:
            arr (array):
                The data values associated with the grid
            idx (tuple):
                The index of the point to evaluate. This is a tuple of length
                `grid.num_axes` with the either -1 or `dim` as the entry for the
                axis associated with this boundary condition. Here, `dim` is the
                dimension of the axis. The index is optional if dim == 1.

        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                raise ValueError(
                    "Index `idx` can only be deduced for grids with a single axis."
                )

        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)

        # calculate necessary constants
        data = self.get_virtual_point_data()

        if self.homogeneous:
            return (  # type: ignore
                data[0]
                + data[1] * arr_1d[..., data[2]]
                + data[3] * arr_1d[..., data[4]]
            )
        else:
            return (  # type: ignore
                data[0][bc_idx]
                + data[1][bc_idx] * arr_1d[..., data[2]]
                + data[3][bc_idx] * arr_1d[..., data[4]]
            )

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        """returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """
        normal = self.normal
        axis = self.axis
        size = self.grid.shape[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        if size < 2:
            raise ValueError(
                f"Need two support points along axis {self.axis} to apply conditions"
            )

        # calculate necessary constants
        data = self.get_virtual_point_data()

        if self.homogeneous:

            @register_jitable
            def virtual_point(arr: np.ndarray, idx: Tuple[int, ...], args=None):
                """evaluate the virtual point at `idx`"""
                arr_1d, _, _ = get_arr_1d(arr, idx)
                if normal:
                    val1 = arr_1d[..., axis, data[2]]
                    val2 = arr_1d[..., axis, data[4]]
                else:
                    val1 = arr_1d[..., data[2]]
                    val2 = arr_1d[..., data[4]]
                return data[0] + data[1] * val1 + data[3] * val2

        else:

            @register_jitable
            def virtual_point(arr: np.ndarray, idx: Tuple[int, ...], args=None):
                """evaluate the virtual point at `idx`"""
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                if normal:
                    val1 = arr_1d[..., axis, data[2]]
                    val2 = arr_1d[..., axis, data[4]]
                else:
                    val1 = arr_1d[..., data[2]]
                    val2 = arr_1d[..., data[4]]
                return data[0][bc_idx] + data[1][bc_idx] * val1 + data[3][bc_idx] * val2

        return virtual_point  # type: ignore

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        """returns a function evaluating the value adjacent to a given point

        Returns:
            function: A function with signature (arr_1d, i_point, bc_idx), where
            `arr_1d` is the one-dimensional data array (the data points along
            the axis perpendicular to the boundary), `i_point` is the index into
            this array for the current point and bc_idx are the remaining
            indices of the current point, which indicate the location on the
            boundary plane. The result of the function is the data value at the
            adjacent point along the axis associated with this boundary
            condition in the upper (lower) direction when `upper` is True
            (False).
        """
        size = self.grid.shape[self.axis]
        if size < 2:
            raise ValueError(
                f"Need at least two support points along axis {self.axis} to apply "
                "boundary conditions"
            )

        # get values distinguishing upper from lower boundary
        if self.upper:
            i_bndry = size - 1
            i_dx = 1
        else:
            i_bndry = 0
            i_dx = -1

        # calculate necessary constants
        data_vp = self.get_virtual_point_data()

        zeros = np.zeros_like(self.value)
        ones = np.ones_like(self.value)

        if self.homogeneous:
            # the boundary condition does not depend on space

            @register_jitable
            def adjacent_point(
                arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
            ) -> float:
                """evaluate the value adjacent to the current point"""
                # determine the parameters for evaluating adjacent point
                if i_point == i_bndry:
                    data = data_vp
                else:
                    data = (zeros, ones, i_point + i_dx, zeros, 0)

                # calculate the values
                return (  # type: ignore
                    data[0]
                    + data[1] * arr_1d[..., data[2]]
                    + data[3] * arr_1d[..., data[4]]
                )

        else:
            # the boundary condition is a function of space

            @register_jitable
            def adjacent_point(
                arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
            ) -> float:
                """evaluate the value adjacent to the current point"""
                # determine the parameters for evaluating adjacent point
                if i_point == i_bndry:
                    data = data_vp
                else:
                    data = (zeros, ones, i_point + i_dx, zeros, 0)

                return (  # type: ignore
                    data[0][bc_idx]
                    + data[1][bc_idx] * arr_1d[..., data[2]]
                    + data[3][bc_idx] * arr_1d[..., data[4]]
                )

        return adjacent_point  # type: ignore

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for this boundary

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args:
                Additional arguments that might be supported by special boundary
                conditions.
        """
        # calculate necessary constants
        data = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: List[Union[slice, int]] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_1 = idx_write[:]
        idx_1[offset + self.axis] = data[2] + 1
        idx_2 = idx_write[:]
        idx_2[offset + self.axis] = data[4] + 1

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_1[offset - 1] = self.axis
            idx_2[offset - 1] = self.axis

        # add dimension to const until it can be broadcasted to shape of data_full
        const, factor1, factor2 = data[0], data[1], data[3]
        while factor1.ndim < self.grid.num_axes:
            factor1 = factor1[..., np.newaxis]
        while factor2.ndim < self.grid.num_axes:
            factor2 = factor2[..., np.newaxis]
        while const.ndim < self.grid.num_axes:
            const = const[..., np.newaxis]

        # calculate the virtual points
        data_full[tuple(idx_write)] = (
            const
            + factor1 * data_full[tuple(idx_1)]
            + factor2 * data_full[tuple(idx_2)]
        )


class CurvatureBC(ConstBC2ndOrderBase):
    """represents a boundary condition imposing the 2nd normal derivative at the
    boundary"""

    names = ["curvature", "second_derivative", "extrapolate"]  # identifiers for this BC
    normal = False

    def get_virtual_point_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
        """return data suitable for calculating virtual points

        Returns:
            tuple: the data structure associated with this virtual point
        """
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]

        if size < 2:
            raise RuntimeError(
                "Need at least 2 support points to use curvature boundary condition"
            )

        value = np.asarray(self.value * dx**2)
        f1 = np.full_like(value, 2.0)
        f2 = np.full_like(value, -1.0)
        if self.upper:
            i1, i2 = size - 1, size - 2
        else:
            i1, i2 = 0, 1
        return (value, f1, i1, f2, i2)


class CurvatureNormalBC(CurvatureBC):
    """represents a boundary condition imposing the 2nd normal derivative at the
    boundary only on the normal component of the tensor field"""

    names = ["curvature_normal"]
    normal = True


def registered_boundary_condition_classes() -> Dict[str, Type[BCBase]]:
    """returns all boundary condition classes that are currently defined

    Returns:
        dict: a dictionary with the names of the boundary condition classes
    """
    return {
        cls_name: cls
        for cls_name, cls in BCBase._subclasses.items()
        if not ("Base" in cls_name or cls_name.startswith("_"))  # skip internal classes
    }


def registered_boundary_condition_names() -> Dict[str, Type[BCBase]]:
    """returns all named boundary conditions that are currently defined

    Returns:
        dict: a dictionary with the names of the boundary conditions that can be used
    """
    return {cls_name: cls for cls_name, cls in BCBase._conditions.items()}
