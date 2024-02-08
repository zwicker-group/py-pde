r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module contains classes for handling a single boundary of a non-periodic axis.
Since an axis has two boundary, we simply distinguish them by a boolean flag `upper`,
which is `True` for the side of the axis with the larger coordinate.

The module currently supports the following standard boundary conditions:

* :class:`~pde.grids.boundaries.local.DirichletBC`:
  Imposing the value of a field at the boundary
* :class:`~pde.grids.boundaries.local.NeumannBC`:
  Imposing the derivative of a field in the outward normal direction at the boundary
* :class:`~pde.grids.boundaries.local.MixedBC`:
  Imposing the derivative of a field in the outward normal direction proportional to its
  value at the boundary  
* :class:`~pde.grids.boundaries.local.CurvatureBC`:
  Imposing the second derivative (curvature) of a field at the boundary

There are also variants of these boundary conditions that only affect the normal
components of a vector or tensor field:
:class:`~pde.grids.boundaries.local.NormalDirichletBC`,
:class:`~pde.grids.boundaries.local.NormalNeumannBC`,
:class:`~pde.grids.boundaries.local.NormalMixedBC`, and
:class:`~pde.grids.boundaries.local.NormalCurvatureBC`.

Finally, there are more specialized classes, which offer greater flexibility, but might
also require a slightly deeper understanding for proper use:

* :class:`~pde.grids.boundaries.local.ExpressionValueBC`:
  Imposing the value of a field at the boundary based on a mathematical expression or a
  python function.
* :class:`~pde.grids.boundaries.local.ExpressionDerivativeBC`:
  Imposing the derivative of a field in the outward normal direction at the boundary
  based on a mathematical expression or a python function.
* :class:`~pde.grids.boundaries.local.ExpressionMixedBC`:
  Imposing a mixed (Robin) boundary condition using mathematical expressions or python
  functions.
* :class:`~pde.grids.boundaries.local.UserBC`:
  Allows full control for setting virtual points, values, or derivatives. The boundary
  conditions are never enforced automatically. It is thus the user's responsibility to
  ensure virtual points are set correctly before operators are applied. To set boundary
  conditions a dictionary :code:`{TARGET: value}` must be supplied as argument `args` to
  :meth:`set_ghost_cells` or the numba equivalent. Here, `TARGET` determines how the
  `value` is interpreted and what boundary condition is actually enforced: the value of
  the virtual points directly (`virtual_point`), the value of the field at the boundary
  (`value`) or the outward derivative of the field at the boundary (`derivative`).

Note that derivatives are generally given in the direction of the outward normal vector,
such that positive derivatives correspond to a function that increases across the
boundary.


**Inheritance structure of the classes:**

.. inheritance-diagram:: pde.grids.boundaries.local
   :parts: 1
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

from ...tools.cache import cached_method
from ...tools.docstrings import fill_in_docstring
from ...tools.numba import address_as_void_pointer, jit, numba_dict
from ...tools.typing import (
    AdjacentEvaluator,
    FloatNumerical,
    GhostCellSetter,
    VirtualPointEvaluator,
)
from ..base import GridBase, PeriodicityError

if TYPE_CHECKING:
    from .._mesh import GridMesh


BoundaryData = Union[dict, str, "BCBase"]


class BCDataError(ValueError):
    """exception that signals that incompatible data was supplied for the BC"""


def _get_arr_1d(arr, idx: tuple[int, ...], axis: int) -> tuple[np.ndarray, int, tuple]:
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
        bc_idx: tuple = (...,)
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
) -> Callable[[np.ndarray, tuple[int, ...]], tuple[np.ndarray, int, tuple]]:
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
    ResultType = tuple[np.ndarray, int, tuple]

    # extract the correct indices
    if dim == 1:

        def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
            """extract the 1d array along axis at point idx"""
            i = idx[0]
            bc_idx: tuple = (...,)
            arr_1d = arr
            return arr_1d, i, bc_idx

    elif dim == 2:
        if axis == 0:

            def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                i, y = idx
                bc_idx = (..., y)
                arr_1d = arr[..., :, y]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, i = idx
                bc_idx = (..., x)
                arr_1d = arr[..., x, :]
                return arr_1d, i, bc_idx

    elif dim == 3:
        if axis == 0:

            def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                i, y, z = idx
                bc_idx = (..., y, z)
                arr_1d = arr[..., :, y, z]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, i, z = idx
                bc_idx = (..., x, z)
                arr_1d = arr[..., x, :, z]
                return arr_1d, i, bc_idx

        elif axis == 2:

            def get_arr_1d(arr: np.ndarray, idx: tuple[int, ...]) -> ResultType:
                """extract the 1d array along axis at point idx"""
                x, y, i = idx
                bc_idx = (..., x, y)
                arr_1d = arr[..., x, y, :]
                return arr_1d, i, bc_idx

    else:
        raise NotImplementedError

    return register_jitable(inline="always")(get_arr_1d)  # type: ignore


TBC = TypeVar("TBC", bound="BCBase")


class BCBase(metaclass=ABCMeta):
    """represents a single boundary in an BoundaryPair instance"""

    names: list[str]
    """list: identifiers used to specify the given boundary class"""
    homogeneous: bool
    """bool: determines whether the boundary condition depends on space"""
    normal: bool = False
    """bool: determines whether the boundary condition only affects normal components.
    
    If this flag is `False`, boundary conditions must specify values for all components
    of the field. If `True`, only the normal components at the boundary are specified.
    """
    _axes_indices: str = "αβγδ"
    """ str: indices used to indicate arbitrary axes in boundary conditions"""

    _subclasses: dict[str, type[BCBase]] = {}  # all classes inheriting from this
    _conditions: dict[str, type[BCBase]] = {}  # mapping from all names to classes

    def __init__(self, grid: GridBase, axis: int, upper: bool, *, rank: int = 0):
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
        """register all subclasses to reconstruct them later"""
        super().__init_subclass__(**kwargs)

        if cls is not BCBase:
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}")
            cls._subclasses[cls.__name__] = cls

            if hasattr(cls, "names"):
                for name in cls.names:
                    cls._conditions[name] = cls

    @property
    def periodic(self) -> bool:
        """bool: whether the boundary condition is periodic"""
        # we determine the periodicity of the boundary condition from the condition
        # itself so we can check for consistency against the grid periodicity
        return isinstance(self, _PeriodicBC)

    @property
    def axis_coord(self) -> float:
        """float: value of the coordinate that defines this boundary condition"""
        if self.upper:
            return self.grid.axes_bounds[self.axis][1]
        else:
            return self.grid.axes_bounds[self.axis][0]

    def _field_repr(self, field_name: str) -> str:
        """return representation of the field to which the condition is applied

        Args:
            field_name (str): Symbol of the field variable

        Returns:
            str: A field with indices denoting which components will be modified
        """
        axis_name = self.grid.axes[self.axis]
        if self.normal:
            assert self.rank > 0
            field_indices = self._axes_indices[: self.rank - 1] + axis_name
        else:
            field_indices = self._axes_indices[: self.rank]
        if field_indices:
            return f"{field_name}_{field_indices}"
        else:
            return f"{field_name}"

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        raise NotImplementedError

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

    def _repr_value(self) -> list[str]:
        return []

    def __repr__(self):
        args = [f"grid={self.grid}", f"axis={self.axis}", f"upper={self.upper}"]
        if self.rank != 0:
            args.append(f"rank={self.rank}")
        args += self._repr_value()
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __str__(self):
        args = [f"axis={self.axis}", "upper" if self.upper else "lower"]
        if self.rank != 0:
            args.append(f"rank={self.rank}")
        args += self._repr_value()
        return f"{self.__class__.__name__}({', '.join(args)})"

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
            and self.normal == other.normal
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
        *,
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
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        data: dict[str, Any],
        *,
        rank: int = 0,
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
            return cls.from_str(grid, axis, upper, condition=b_type, rank=rank, **data)

        elif len(data) == 1:
            # only a single items is given
            b_type, b_value = data.popitem()
            return cls.from_str(
                grid, axis, upper, condition=b_type, rank=rank, value=b_value, **data
            )

        else:
            raise BCDataError(
                f"Boundary conditions `{str(list(data.keys()))}` are not supported."
            )

    @classmethod
    def from_data(
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        data: BoundaryData,
        *,
        rank: int = 0,
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
            if data.grid._mesh is not None:
                # we need to exclude this case since otherwise we get into a rabit hole
                # where it is not clear what grid boundary conditions belong to. The
                # idea is that users only create boundary conditions for the full grid
                # and that the splitting onto subgrids is only done once, automatically,
                # and without involving calls to `from_data`
                raise ValueError("Cannot create MPI subgrid BC from data")

            if data.grid != grid or data.axis != axis or data.rank != rank:
                raise ValueError(f"Incompatible: {data!r} & {grid=}, {axis=}, {rank=})")
            bc = data.copy(upper=upper)

        elif isinstance(data, dict):
            # create from dictionary
            bc = cls.from_dict(grid, axis, upper=upper, data=data, rank=rank)

        elif isinstance(data, str):
            # create a specific condition given by a string
            bc = cls.from_str(grid, axis, upper=upper, condition=data, rank=rank)

        else:
            raise BCDataError(f"Unsupported format: `{data}`. " + cls.get_help())

        # check consistency
        if bc.periodic != grid.periodic[axis]:
            raise PeriodicityError("Periodicity of conditions must match grid")
        return bc

    def to_subgrid(self: TBC, subgrid: GridBase) -> TBC:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        raise NotImplementedError("Boundary condition cannot be transfered to subgrid")

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

    def copy(self: TBC, upper: bool | None = None, rank: int | None = None) -> TBC:
        raise NotImplementedError

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        """returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        """returns a function evaluating the value adjacent to a given point

        .. deprecated:: Since 2023-12-19

        Returns:
            function: A function with signature (arr_1d, i_point, bc_idx), where
            `arr_1d` is the one-dimensional data array (the data points along the axis
            perpendicular to the boundary), `i_point` is the index into this array for
            the current point and bc_idx are the remaining indices of the current point,
            which indicate the location on the boundary plane. The result of the
            function is the data value at the adjacent point along the axis associated
            with this boundary condition in the upper (lower) direction when `upper` is
            True (False).
        """
        raise NotImplementedError

    @abstractmethod
    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """set the ghost cell values for this boundary

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args (:class:`~numpy.ndarray`):
                Determines what boundary conditions are set. `args` should be set to
                :code:`{TARGET: value}`. Here, `TARGET` determines how the `value` is
                interpreted and what boundary condition is actually enforced: the value
                of the virtual points directly (`virtual_point`), the value of the field
                at the boundary (`value`) or the outward derivative of the field at the
                boundary (`derivative`).
        """

    def make_ghost_cell_sender(self) -> GhostCellSetter:
        """return function that might mpi_send data to set ghost cells for this boundary"""

        @register_jitable
        def noop(data_full: np.ndarray, args=None) -> None:
            """no-operation as the default case"""

        return noop  # type: ignore

    def _get_value_cell_index(self, with_ghost_cells: bool) -> int:
        """determine index of the cell from which field value is read

        Args:
            with_ghost_cells (bool):
                Determines whether the index is supposed to be into an array with ghost
                cells or not
        """
        if self.upper:
            if with_ghost_cells:
                return self.grid.shape[self.axis]
            else:
                return self.grid.shape[self.axis] - 1
        else:
            if with_ghost_cells:
                return 1
            else:
                return 0

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells for this boundary"""
        normal = self.normal
        axis = self.axis

        # get information of the virtual points (ghost cells)
        vp_idx = self.grid.shape[self.axis] + 1 if self.upper else 0
        np_idx = self._get_value_cell_index(with_ghost_cells=False)
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


class _MPIBC(BCBase):
    """represents a boundary that is exchanged with another MPI process"""

    homogeneous = False

    def __init__(
        self,
        mesh: GridMesh,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        node_id: int | None = None,
    ):
        """
        Args:
            mesh (:class:`~pde.grids._mesh.GridMesh`):
                Grid mesh describing the distributed MPI nodes
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            node_id (int):
                The MPI node (the subgrid ID) this BC is associated with
        """
        super().__init__(mesh[node_id], axis, upper, rank=rank)
        neighbor_id = mesh.get_neighbor(axis, upper, node_id=node_id)
        if neighbor_id is None:
            raise RuntimeError("No neighboring cell for this boundary")
        self._neighbor_id = neighbor_id
        self._mpi_flag = mesh.get_boundary_flag(self._neighbor_id, upper)

        # determine indices for reading and writing data
        idx: list[Any] = [slice(1, -1)] * self.grid.num_axes
        idx[self.axis] = -2 if self.upper else 1  # read valid data
        self._idx_read = tuple([Ellipsis] + idx)
        idx[self.axis] = -1 if self.upper else 0  # write ghost cells
        self._idx_write = tuple([Ellipsis] + idx)

    def _repr_value(self):
        return [f"neighbor={self._neighbor_id}"]

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.grid == other.grid
            and self.axis == other.axis
            and self.rank == other.rank
            and self._neighbor_id == other._neighbor_id
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        axis_name = self.grid.axes[self.axis]
        return f"MPI @ {axis_name}={self.axis_coord}"

    def send_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        """mpi_send the ghost cell values for this boundary

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
        """
        from ...tools.mpi import mpi_send

        mpi_send(data_full[self._idx_read], self._neighbor_id, self._mpi_flag)

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        from ...tools.mpi import mpi_recv

        mpi_recv(data_full[self._idx_write], self._neighbor_id, self._mpi_flag)

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        raise NotImplementedError

    def make_ghost_cell_sender(self) -> GhostCellSetter:
        """return function that sends data to set ghost cells for other boundaries"""
        from ...tools.mpi import mpi_send

        cell = self._neighbor_id
        flag = self._mpi_flag
        num_axes = self.grid.num_axes
        axis = self.axis
        idx = -2 if self.upper else 1  # index for reading data

        if num_axes == 1:

            def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                mpi_send(data_full[..., idx], cell, flag)

        elif num_axes == 2:
            if axis == 0:

                def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                    mpi_send(data_full[..., idx, 1:-1], cell, flag)

            else:

                def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, idx], cell, flag)

        elif num_axes == 3:
            if axis == 0:

                def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                    mpi_send(data_full[..., idx, 1:-1, 1:-1], cell, flag)

            elif axis == 1:

                def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, idx, 1:-1], cell, flag)

            else:

                def ghost_cell_sender(data_full: np.ndarray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, 1:-1, idx], cell, flag)

        else:
            raise NotImplementedError

        return register_jitable(ghost_cell_sender)  # type: ignore

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells for this boundary"""
        from ...tools.mpi import mpi_recv

        cell = self._neighbor_id
        flag = self._mpi_flag
        num_axes = self.grid.num_axes
        axis = self.axis
        idx = -1 if self.upper else 0  # index for writing data

        if num_axes == 1:

            def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                if data_full.ndim == 1:
                    # in this case, `data_full[..., idx]` is a scalar, which numba
                    # treats differently, so `numba_mpi.mpi_recv` fails
                    buffer = np.empty((), dtype=data_full.dtype)
                    mpi_recv(buffer, cell, flag)
                    data_full[..., idx] = buffer
                else:
                    mpi_recv(data_full[..., idx], cell, flag)

        elif num_axes == 2:
            if axis == 0:

                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    mpi_recv(data_full[..., idx, 1:-1], cell, flag)

            else:

                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, idx], cell, flag)

        elif num_axes == 3:
            if axis == 0:

                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    mpi_recv(data_full[..., idx, 1:-1, 1:-1], cell, flag)

            elif axis == 1:

                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, idx, 1:-1], cell, flag)

            else:

                def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, 1:-1, idx], cell, flag)

        else:
            raise NotImplementedError

        return register_jitable(ghost_cell_setter)  # type: ignore


class UserBC(BCBase):
    """represents a boundary whose virtual point are set by the user.

    Boundary conditions will only be set when a dictionary :code:`{TARGET: value}` is
    supplied as argument `args` to :meth:`set_ghost_cells` or the numba equivalent.
    Here, `TARGET` determines how the `value` is interpreted and what boundary condition
    is actually enforced: the value of the virtual points directly (`virtual_point`),
    the value of the field at the boundary (`value`) or the outward derivative of the
    field at the boundary (`derivative`).

    Warning:
        This implies that the boundary conditions are never enforced automatically,
        e.g., when evaluating an operator. It is thus the user's responsibility to
        ensure virtual points are set correctly before operators are applied.
    """

    names = ["user"]

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        axis_name = self.grid.axes[self.axis]
        return f"user-controlled  @ {axis_name}={self.axis_coord}"

    def copy(self: TBC, upper: bool | None = None, rank: int | None = None) -> TBC:
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
        )

    def to_subgrid(self: TBC, subgrid: GridBase) -> TBC:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        return self.__class__(
            grid=subgrid, axis=self.axis, upper=self.upper, rank=self.rank
        )

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        if args is None:
            # usual case where set_ghost_cells is called automatically. In our case,
            # won't do anything since we expect the user to call the function manually
            # with the user data provided as the argument.
            return

        if any(t in args for t in ["virtual_point", "value", "derivative"]):
            # ghost cells will only be set if any of the above keys were supplied

            # prepare the array of slices to index bcs
            offset = data_full.ndim - self.grid.num_axes  # additional data axes
            idx_offset = [slice(None)] * offset
            idx_valid = [slice(1, -1)] * self.grid.num_axes
            idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
            idx_write[offset + self.axis] = -1 if self.upper else 0
            idx_read = idx_write[:]
            idx_read[offset + self.axis] = -2 if self.upper else 1

            if self.normal:
                assert offset > 0
                idx_write[offset - 1] = self.axis
                idx_read[offset - 1] = self.axis

            # get values right next to the boundary
            bndry_values = data_full[tuple(idx_read)]

            # calculate the virtual points
            if "virtual_point" in args:
                data_full[tuple(idx_write)] = args["virtual_point"]
            elif "value" in args:
                data_full[tuple(idx_write)] = 2 * args["value"] - bndry_values
            elif "derivative" in args:
                dx = self.grid.discretization[self.axis]
                data_full[tuple(idx_write)] = dx * args["derivative"] + bndry_values
            else:
                raise RuntimeError
        # else: no-op for the default case where BCs are not set by user

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        dx = self.grid.discretization[self.axis]

        def extract_value(values, arr: np.ndarray, idx: tuple[int, ...]):
            """helper function that extracts the correct value from supplied ones"""
            if isinstance(values, (nb.types.Number, Number)):
                # scalar was supplied => simply return it
                return values
            elif isinstance(arr, (nb.types.Array, np.ndarray)):
                # array was supplied => extract value at current position
                _, _, bc_idx = get_arr_1d(arr, idx)
                return values[bc_idx]
            else:
                raise TypeError("Either a scalar or an array must be supplied")

        @overload(extract_value)
        def ol_extract_value(values, arr: np.ndarray, idx: tuple[int, ...]):
            """helper function that extracts the correct value from supplied ones"""
            if isinstance(values, (nb.types.Number, Number)):
                # scalar was supplied => simply return it
                def impl(values, arr: np.ndarray, idx: tuple[int, ...]):
                    return values

            elif isinstance(arr, (nb.types.Array, np.ndarray)):
                # array was supplied => extract value at current position

                def impl(values, arr: np.ndarray, idx: tuple[int, ...]):
                    _, _, bc_idx = get_arr_1d(arr, idx)
                    return values[bc_idx]

            else:
                raise TypeError("Either a scalar or an array must be supplied")

            return impl

        @register_jitable
        def virtual_point(arr: np.ndarray, idx: tuple[int, ...], args):
            """evaluate the virtual point at `idx`"""
            if "virtual_point" in args:
                # set the virtual point directly
                return extract_value(args["virtual_point"], arr, idx)

            elif "value" in args:
                # set the value at the boundary
                value = extract_value(args["value"], arr, idx)
                return 2 * value - arr[idx]

            elif "derivative" in args:
                # set the outward derivative at the boundary
                value = extract_value(args["derivative"], arr, idx)
                return dx * value + arr[idx]

            else:
                # no-op for the default case where BCs are not set by user
                return math.nan

        return virtual_point  # type: ignore

    def make_ghost_cell_setter(self) -> GhostCellSetter:
        """return function that sets the ghost cells for this boundary"""
        ghost_cell_setter_inner = super().make_ghost_cell_setter()

        @register_jitable
        def ghost_cell_setter(data_full: np.ndarray, args=None) -> None:
            """helper function setting the conditions on all axes"""
            if args is None:
                return  # no-op when no specific arguments are given

            if "virtual_point" in args or "value" in args or "derivative" in args:
                # ghost cells will only be set if any of the above keys were supplied
                ghost_cell_setter_inner(data_full, args=args)
            # else: no-op for the default case where BCs are not set by user

        return ghost_cell_setter  # type: ignore


ExpressionBCTargetType = Literal["value", "derivative", "mixed", "virtual_point"]


class ExpressionBC(BCBase):
    """represents a boundary whose virtual point is calculated from an expression

    The expression is given as a string and will be parsed by :mod:`sympy` or a function
    that is optionally compiled with :mod:`numba`. The expression can contain typical
    mathematical operators and may depend on the value at the last support point next to
    the boundary (`value`), spatial coordinates defined by the grid marking the boundary
    point (e.g., `x` or `r`), and time `t`.
    """

    names = ["virtual_point"]

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        const: float | str | Callable = 0,
        target: ExpressionBCTargetType = "virtual_point",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        r"""
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
            value (float or str or callable):
                An expression that determines the value of the boundary condition.
                Alternatively, this can be a function with signature `(value, dx,
                *coords, t)` that determines the value of `target` from the field value
                `value` (the value of the adjacent cell unless `value_cell` is
                specified), the spatial discretization `dx` in the direction
                perpendicular to the wall, the spatial coordinates of the wall point,
                and time `t`. Ideally, this function should be numba-compilable since
                simulations might otherwise be very slow.
            const (float or str or callable):
                An expression similar to `value`, which is only used for mixed (Robin)
                boundary conditions. Note that the implementation currently does not
                support that one argument is given as a callable function while the
                other is defined via an expression, so both need to have the same type.
            target (str):
                Selects which value is actually set. Possible choices include `value`,
                `derivative`, `mixed`, and `virtual_point`.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in expressions
            value_cell (int):
                Determines which cells is read to determine the field value that is used
                as `value` in the expression or the function call. The default (`None`)
                specifies the adjacent cell.
        """
        super().__init__(grid, axis, upper, rank=rank)
        self.value_cell = value_cell

        if self.rank != 0:
            raise NotImplementedError(
                "Expression boundary conditions only work for scalar conditions"
            )

        # store data for later use
        self._input: dict[str, Any] = {
            "value_expr": value,
            "const_expr": const,
            "target": target,
            "user_funcs": user_funcs,
        }
        signature = ["value", "dx"] + grid.axes + ["t"]

        if callable(value) or callable(const):
            # the coefficients are given as functions
            self._is_func = True
        else:
            # the coefficients are expressions or constant values
            self._is_func = False
            if target == "virtual_point":
                expression = value
            elif target == "value":
                # Dirichlet boundary condition
                expression = f"2 * ({value}) - value"
            elif target == "derivative":
                # Neumann boundary condition
                expression = f"dx * ({value}) + value"
            elif target == "mixed":
                # special case of a Robin boundary condition, which also uses `const`
                enumerator = f"2 * dx * ({const}) + (2 - ({value}) * dx) * value"
                expression = f"({enumerator}) / (({value}) * dx + 2)"
            else:
                raise ValueError(f"Unknown target `{target}` for expression")

            # parse this expression
            from pde.tools.expressions import ScalarExpression

            self._func_expression = ScalarExpression(
                expression,
                signature=signature,
                user_funcs=user_funcs,
                repl=grid.c._axes_alt_repl,
            )

        # quickly check whether the expression was parsed correctly
        try:
            self._func(do_jit=False)(*self._test_values)
        except Exception as err:
            if self._is_func:
                raise BCDataError(
                    f"Could not evaluate BC function. Expected signature "
                    f"{signature}.\nEncountered error: {err}"
                )
            else:
                raise BCDataError(
                    f"Could not evaluate BC expression `{expression}` with signature "
                    f"{signature}.\nEncountered error: {err}"
                )

    @property
    def _test_values(self) -> tuple[float, ...]:
        """tuple: suitable values with which the user expression can be tested"""
        test_values = [
            np.zeros((self.grid.dim,) * self.rank),
            self.grid.discretization[self.axis],
        ]
        bc_coords = self.grid._boundary_coordinates(self.axis, self.upper)
        test_values.extend(np.moveaxis(bc_coords, -1, 0))
        test_values.append(0)
        return tuple(test_values)

    def _prepare_function(self, func: Callable | float, do_jit: bool) -> Callable:
        """helper function that compiles a single function given as a parameter"""
        if not callable(func):
            # the function is just a number, which we also support
            func_value = float(func)  # TODO: support complex numbers

            @register_jitable
            def value_func(*args):
                return func_value

            return value_func  # type: ignore

        elif not do_jit:
            # function is callable, but does not need to be compiled
            return func

        else:
            # function is callable and needs to be compiled
            try:
                # try compiling the function
                value_func = jit(func)
                # and evaluate it, so compilation is forced
                value_func(*self._test_values)

                if os.environ.get("PYPDE_TESTRUN"):
                    # ensure that the except path is also tested
                    raise nb.NumbaError("Force except")

            except nb.NumbaError:
                # if compilation fails, we simply fall back to pure-python mode
                self._logger.warning(f"Cannot compile BC {self}")

                @register_jitable
                def value_func(*args):
                    with nb.objmode(value="double"):
                        value = func(*args)
                    return value

            return value_func  # type: ignore

    def _get_function_from_userfunc(self, do_jit: bool) -> Callable:
        """returns function from user function evaluating the value of the virtual point

        Args:
            do_jit (bool):
                Determines whether the returned function is numba-compiled
        """
        # `value` is a callable function
        target = self._input["target"]
        value_func = self._prepare_function(self._input["value_expr"], do_jit=do_jit)

        if target == "virtual_point":
            return value_func

        elif target == "value":
            # Dirichlet boundary condition

            @register_jitable
            def virtual_from_value(adjacent_value, *args):
                return 2 * value_func(adjacent_value, *args) - adjacent_value

            return virtual_from_value  # type: ignore

        elif target == "derivative":
            # Neumann boundary condition

            @register_jitable
            def virtual_from_derivative(adjacent_value, dx, *args):
                return dx * value_func(adjacent_value, dx, *args) + adjacent_value

            return virtual_from_derivative  # type: ignore

        elif target == "mixed":
            # special case of a Robin boundary condition, which also uses `const`
            const_func = self._prepare_function(
                self._input["const_expr"], do_jit=do_jit
            )

            @register_jitable
            def virtual_from_mixed(adjacent_value, dx, *args):
                value_dx = dx * value_func(adjacent_value, dx, *args)
                const_value = const_func(adjacent_value, dx, *args)
                expr_A = 2 * dx / (value_dx + 2) * const_value
                expr_B = (value_dx - 2) / (value_dx + 2)
                return expr_A - expr_B * adjacent_value

            return virtual_from_mixed  # type: ignore

        else:
            raise ValueError(f"Unknown target `{target}` for expression")

    def _get_function_from_expression(self, do_jit: bool) -> Callable:
        """returns function from expression evaluating the value of the virtual point

        Args:
            do_jit (bool):
                Determines whether the returned function is numba-compiled
        """
        if not do_jit:
            return self._func_expression

        func = self._func_expression._get_function_cached(single_arg=False)
        try:
            # try to compile the expression that was given
            value_func = jit(func)
            # call the function to actually trigger compilation
            value_func(*self._test_values)

            if os.environ.get("PYPDE_TESTRUN"):
                # ensure that the except path is also tested
                raise nb.NumbaError("Force except")

        except nb.NumbaError:
            # if compilation fails, we simply fall back to pure-python mode
            self._logger.warning(f"Cannot compile BC {self._func_expression}")
            # calculate the expected value to test this later (and fail early)
            expected = func(*self._test_values)

            num_axes = self.grid.num_axes
            if num_axes == 1:

                @jit
                def value_func(grid_value, dx, x, t):
                    with nb.objmode(value="double"):
                        value = func(grid_value, dx, x, t)
                    return value

            elif num_axes == 2:

                @jit
                def value_func(grid_value, dx, x, y, t):
                    with nb.objmode(value="double"):
                        value = func(grid_value, dx, x, y, t)
                    return value

            elif num_axes == 3:

                @jit
                def value_func(grid_value, dx, x, y, z, t):
                    with nb.objmode(value="double"):
                        value = func(grid_value, dx, x, y, z, t)
                    return value

            else:
                # cheap way to signal a problem
                raise ValueError

            # compile the actual functio and check the result
            result_compiled = value_func(*self._test_values)
            if not np.allclose(result_compiled, expected):
                raise RuntimeError("Compiled function does not give same value")

        return value_func  # type: ignore

    @cached_method()
    def _func(self, do_jit: bool) -> Callable:
        """returns function that evaluates the value of the virtual point

        Args:
            do_jit (bool):
                Determines whether the returned function is numba-compiled
        """
        if self._is_func:
            return self._get_function_from_userfunc(do_jit=do_jit)
        else:
            return self._get_function_from_expression(do_jit=do_jit)

    def _repr_value(self):
        if self._input["target"] == "mixed":
            # treat the mixed case separately
            res = [
                f'target="{self._input["target"]}", '
                f'value="{self._input["value_expr"]}", '
                f'const="{self._input["const_expr"]}"'
            ]
        elif self._is_func:
            res = [f'{self._input["target"]}=<function>']
        else:
            res = [f'{self._input["target"]}="{self._input["value_expr"]}"']
        if self.value_cell is not None:
            res.append(f", value_cell={self.value_cell}")
        return res

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        axis_name = self.grid.axes[self.axis]
        target = self._input["target"]

        if self._is_func:
            value_expr = "<function>"
        else:
            value_expr = self._input["value_expr"]
        const_expr = self._input["const_expr"]

        field = self._field_repr(field_name)
        if target == "virtual_point":
            return f"{field} = {value_expr}   @ virtual point"
        elif target == "value":
            return f"{field} = {value_expr}   @ {axis_name}={self.axis_coord}"
        elif target == "derivative":
            sign = " " if self.upper else "-"
            return f"{sign}∂{field}/∂{axis_name} = {value_expr}   @ {axis_name}={self.axis_coord}"
        elif target == "mixed":
            sign = " " if self.upper else "-"
            return (
                f"{sign}∂{field}/∂{axis_name} + ({value_expr})*{field} = "
                f"{const_expr}   @ {axis_name}={self.axis_coord}"
            )
        else:
            raise NotImplementedError(f"Unsupported target `{target}`")

    def __eq__(self, other):
        """checks for equality neglecting the `upper` property"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            super().__eq__(other)
            and self._input == other._input
            and self.value_cell == other.value_cell
        )

    def copy(
        self: ExpressionBC, upper: bool | None = None, rank: int | None = None
    ) -> ExpressionBC:
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self._input["value_expr"],
            const=self._input["const_expr"],
            target=self._input["target"],
            user_funcs=self._input["user_funcs"],
            value_cell=self.value_cell,
        )

    def to_subgrid(self: ExpressionBC, subgrid: GridBase) -> ExpressionBC:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)

        if self.value_cell is not None:
            raise NotImplementedError("Custom value indices are not supported")

        bc = self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self._input["value_expr"],
            const=self._input["const_expr"],
            target=self._input["target"],
            user_funcs=self._input["user_funcs"],
            value_cell=self.value_cell,
        )

        # The following call raise error when `value_cell` index is not in `subgrid`.
        bc._get_value_cell_index(with_ghost_cells=False)
        return bc

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        raise NotImplementedError

    def make_adjacent_evaluator(self) -> AdjacentEvaluator:
        raise NotImplementedError

    def _get_value_cell_index(self, with_ghost_cells: bool) -> int:
        if self.value_cell is None:
            # pick adjacent cell by default
            return super()._get_value_cell_index(with_ghost_cells)
        elif self.value_cell >= 0:
            # positive indexing
            idx = int(self.value_cell)
            if idx >= self.grid.shape[self.axis]:
                size = self.grid.shape[self.axis]
                raise IndexError(f"Index {self.value_cell} out of bounds ({size=})")
            return idx + 1 if with_ghost_cells else idx
        else:  # self.value_cell < 0:
            # negative indexing
            idx = int(self.value_cell)
            if idx < -self.grid.shape[self.axis]:
                size = self.grid.shape[self.axis]
                raise IndexError(f"Index {self.value_cell} out of bounds ({size=})")
            return idx - 1 if with_ghost_cells else idx

    def set_ghost_cells(self, data_full: np.ndarray, *, args=None) -> None:
        dx = self.grid.discretization[self.axis]

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_read = idx_write[:]
        idx_read[offset + self.axis] = self._get_value_cell_index(with_ghost_cells=True)

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_read[offset - 1] = self.axis

        # prepare the arguments
        values = data_full[tuple(idx_read)]
        coords = self.grid._boundary_coordinates(axis=self.axis, upper=self.upper)
        coords = np.moveaxis(coords, -1, 0)  # point coordinates to first axis

        if args is None:
            if not self._is_func and self._func_expression.depends_on("t"):
                raise RuntimeError(
                    "Require value for `t` for time-dependent BC. The value must be "
                    "passed explicitly via `args` when calling a differential operator."
                )
            t = 0.0
        else:
            t = float(args["t"])

        # calculate the virtual points
        data_full[tuple(idx_write)] = self._func(do_jit=False)(values, dx, *coords, t)

    def make_virtual_point_evaluator(self) -> VirtualPointEvaluator:
        dx = self.grid.discretization[self.axis]
        num_axes = self.grid.num_axes
        get_arr_1d = _make_get_arr_1d(num_axes, self.axis)
        bc_coords = self.grid._boundary_coordinates(axis=self.axis, upper=self.upper)
        bc_coords = np.moveaxis(bc_coords, -1, 0)  # point coordinates to first axis
        assert num_axes <= 3

        if self._is_func:
            warn_if_time_not_set = False
        else:
            warn_if_time_not_set = self._func_expression.depends_on("t")
        func = self._func(do_jit=True)

        @jit
        def virtual_point(arr: np.ndarray, idx: tuple[int, ...], args=None) -> float:
            """evaluate the virtual point at `idx`"""
            _, _, bc_idx = get_arr_1d(arr, idx)
            grid_value = arr[idx]
            coords = bc_coords[bc_idx]

            # extract time for handling time-dependent BCs
            if args is None or "t" not in args:
                if warn_if_time_not_set:
                    raise RuntimeError(
                        "Require value for `t` for time-dependent BC. The value must "
                        "be passed explicitly via `args` when calling a differential "
                        "operator."
                    )
                t = 0.0
            else:
                t = float(args["t"])

            if num_axes == 1:
                return func(grid_value, dx, coords[0], t)  # type: ignore
            elif num_axes == 2:
                return func(grid_value, dx, coords[0], coords[1], t)  # type: ignore
            elif num_axes == 3:
                return func(grid_value, dx, coords[0], coords[1], coords[2], t)  # type: ignore
            else:
                # cheap way to signal a problem
                return math.nan

        # evaluate the function to force compilation and catch errors early
        virtual_point(np.zeros([3] * num_axes), (0,) * num_axes, numba_dict({"t": 0.0}))

        return virtual_point  # type: ignore


class ExpressionValueBC(ExpressionBC):
    """represents a boundary whose value is calculated from an expression

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["value_expression", "value_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        target: ExpressionBCTargetType = "value",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


class ExpressionDerivativeBC(ExpressionBC):
    """represents a boundary whose outward derivative is calculated from an expression

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["derivative_expression", "derivative_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        target: ExpressionBCTargetType = "derivative",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


class ExpressionMixedBC(ExpressionBC):
    """represents a boundary whose outward derivative is calculated from an expression

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["mixed_expression", "mixed_expr", "robin_expression", "robin_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        const: float | str | Callable = 0,
        target: ExpressionBCTargetType = "mixed",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            const=const,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


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
        value: float | np.ndarray | str = 0,
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
            normal (bool):
                Flag indicating whether the condition is only applied in the normal
                direction.
            value (float or str or :class:`~numpy.ndarray`):
                a value stored with the boundary condition. The interpretation of this
                value depends on the type of boundary condition. If value is a single
                value (or tensor in case of tensorial boundary conditions), the same
                value is applied to all points. Inhomogeneous boundary conditions are
                possible by supplying an expression as a string, which then may depend
                on the axes names of the respective grid.
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

    @value.setter
    @fill_in_docstring
    def value(self, value: float | np.ndarray | str = 0):
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
    def _parse_value(self, value: float | np.ndarray | str) -> np.ndarray:
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
            expr = ScalarExpression(
                value, self.grid.axes, repl=self.grid.c._axes_alt_repl
            )

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
                coords: dict[str, float] = {name: 0 for name in self.grid.axes}
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
            result = np.broadcast_to(value, self._shape_tensor)

        else:
            # assume tensorial and/or inhomogeneous values
            value = np.asarray(value)

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
                    f"value and its spatial dimensions {self._shape_boundary}. "
                    f"(rank={self.rank}, normal={self.normal})"
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

        return result

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

    def copy(
        self: ConstBCBase,
        upper: bool | None = None,
        rank: int | None = None,
        value: float | np.ndarray | str | None = None,
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

    def to_subgrid(self: ConstBCBase, subgrid: GridBase) -> ConstBCBase:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            raise NotImplementedError("Cannot transfer complicated BC to subgrid")

        return self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self.value,
        )

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

        @nb.njit(nb.typeof(self._value)(), inline="always")
        def get_value() -> np.ndarray:
            """helper function returning the linked array"""
            return nb.carray(address_as_void_pointer(mem_addr), shape, dtype)  # type: ignore

        # keep a reference to the array to prevent garbage collection
        get_value._value_ref = self._value

        return get_value  # type: ignore


class ConstBC1stOrderBase(ConstBCBase):
    """represents a single boundary in an BoundaryPair instance"""

    @abstractmethod
    def get_virtual_point_data(self, compiled: bool = False) -> tuple[Any, Any, int]:
        """return data suitable for calculating virtual points

        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.

        Returns:
            tuple: the data structure associated with this virtual point
        """

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
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

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
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
        normal = self.normal
        axis = self.axis
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)

        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data(compiled=True)

        if self.homogeneous:

            @jit
            def virtual_point(
                arr: np.ndarray, idx: tuple[int, ...], args=None
            ) -> float:
                """evaluate the virtual point at `idx`"""
                arr_1d, _, _ = get_arr_1d(arr, idx)
                if normal:
                    val_field = arr_1d[..., axis, index]
                else:
                    val_field = arr_1d[..., index]
                return const() + factor() * val_field  # type: ignore

        else:

            @jit
            def virtual_point(
                arr: np.ndarray, idx: tuple[int, ...], args=None
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
        # method deprecated since 2023-12-19
        warnings.warn("`make_adjacent_evaluator` is deprecated", DeprecationWarning)
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
                arr_1d: np.ndarray, i_point: int, bc_idx: tuple[int, ...]
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
        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
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

    def __str__(self):
        return '"periodic"'

    def copy(self: _PeriodicBC, upper: bool | None = None) -> _PeriodicBC:  # type: ignore
        """return a copy of itself, but with a reference to the same grid"""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            flip_sign=self.flip_sign,
        )

    def to_subgrid(self: _PeriodicBC, subgrid: GridBase) -> _PeriodicBC:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            raise NotImplementedError("Cannot transfer complicated BC to subgrid")

        return self.__class__(
            grid=subgrid, axis=self.axis, upper=self.upper, flip_sign=self.flip_sign
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        if self.upper:
            other_coord = self.grid.axes_bounds[self.axis][0]
        else:
            other_coord = self.grid.axes_bounds[self.axis][1]

        axis_name = self.grid.axes[self.axis]
        if self.flip_sign:
            return f"{field_name}({axis_name}={self.axis_coord}) = -{field_name}({axis_name}={other_coord})"
        else:
            return f"{field_name}({axis_name}={self.axis_coord}) = {field_name}({axis_name}={other_coord})"

    def get_virtual_point_data(self, compiled: bool = False) -> tuple[Any, Any, int]:
        index = 0 if self.upper else self.grid.shape[self.axis] - 1
        value = -1 if self.flip_sign else 1

        if not compiled:
            return (0.0, value, index)
        else:
            const = np.array(0)
            factor = np.array(value)

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

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        axis_name = self.grid.axes[self.axis]
        field = self._field_repr(field_name)
        return f"{field} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(self, compiled: bool = False) -> tuple[Any, Any, int]:
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

            const = np.array(const)
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


class NeumannBC(ConstBC1stOrderBase):
    """represents a boundary condition imposing the derivative in the outward
    normal direction of the boundary"""

    names = ["derivative", "neumann"]  # identifiers for this boundary condition

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        sign = " " if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        deriv = f"∂{self._field_repr(field_name)}/∂{axis_name}"
        return f"{sign}{deriv} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(self, compiled: bool = False) -> tuple[Any, Any, int]:
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

            const = np.array(const)
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

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | np.ndarray | str = 0,
        const: float | np.ndarray | str = 0,
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
                The parameter :math:`\gamma` quantifying the influence of the field onto
                its normal derivative. If `value` is a single value (or tensor in case
                of tensorial boundary conditions), the same value is applied to all
                points.  Inhomogeneous boundary conditions are possible by supplying an
                expression as a string, which then may depend on the axes names of the
                respective grid.
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

    def copy(
        self: MixedBC,
        upper: bool | None = None,
        rank: int | None = None,
        value: float | np.ndarray | str | None = None,
        const: float | np.ndarray | str | None = None,
    ) -> MixedBC:
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

    def to_subgrid(self: MixedBC, subgrid: GridBase) -> MixedBC:
        """converts this boundary condition to one valid for a given subgrid

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            raise NotImplementedError("Cannot transfer complicated BC to subgrid")

        return self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self.value,
            const=self.const,
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        sign = "" if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        field_repr = self._field_repr(field_name)
        deriv = f"∂{field_repr}/∂{axis_name}"
        return f"{sign}{deriv} + {self.value} * {field_repr} = {self.const}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(self, compiled: bool = False) -> tuple[Any, Any, int]:
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
            const_val = np.array(self.const)
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
            const = np.array(const)
            factor = np.array(factor)

            @register_jitable(inline="always")
            def const_func():
                return const

            @register_jitable(inline="always")
            def factor_func():
                return factor

        return (const_func, factor_func, index)


class ConstBC2ndOrderBase(ConstBCBase):
    """abstract base class for boundary conditions of 2nd order"""

    @abstractmethod
    def get_virtual_point_data(self) -> tuple[Any, Any, int, Any, int]:
        """return data suitable for calculating virtual points

        Returns:
            tuple: the data structure associated with this virtual point
        """

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
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

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
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
                    "Index can only be deduced for grids with a single axis."
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
            def virtual_point(arr: np.ndarray, idx: tuple[int, ...], args=None):
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
            def virtual_point(arr: np.ndarray, idx: tuple[int, ...], args=None):
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
        # method deprecated since 2023-12-19
        warnings.warn("`make_adjacent_evaluator` is deprecated", DeprecationWarning)
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
                arr_1d: np.ndarray, i_point: int, bc_idx: tuple[int, ...]
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
                arr_1d: np.ndarray, i_point: int, bc_idx: tuple[int, ...]
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
        # calculate necessary constants
        data = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
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

        # # add dimension to const until it can be broadcasted to shape of data_full
        const, factor1, factor2 = data[0], data[1], data[3]
        if self.homogeneous:
            if not np.isscalar(factor1):
                for _ in range(self.grid.num_axes - 1):
                    const = const[..., np.newaxis]
            if not np.isscalar(factor1):
                for _ in range(self.grid.num_axes - 1):
                    factor1 = factor1[..., np.newaxis]
            if not np.isscalar(factor2):
                for _ in range(self.grid.num_axes - 1):
                    factor2 = factor2[..., np.newaxis]

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

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """return mathematical representation of the boundary condition"""
        sign = " " if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        deriv = f"∂²{self._field_repr(field_name)}/∂{axis_name}²"
        return f"{sign}{deriv} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
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


class NormalDirichletBC(DirichletBC):
    """represents a boundary condition imposing the value on normal components"""

    names = ["normal_value", "normal_dirichlet", "dirichlet_normal"]
    normal = True


class NormalNeumannBC(NeumannBC):
    """represents a boundary condition imposing the derivative of normal components
    in the outward normal direction of the boundary"""

    names = ["normal_derivative", "normal_neumann", "neumann_normal"]
    normal = True


class NormalMixedBC(MixedBC):
    r"""represents a mixed (or Robin) boundary condition setting the derivative of the
    normal components in the outward normal direction of the boundary using an affine
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

    names = ["normal_mixed", "normal_robin"]
    normal = True


class NormalCurvatureBC(CurvatureBC):
    """represents a boundary condition imposing the 2nd normal derivative onto the
    normal components at the boundary"""

    names = ["normal_curvature"]  # identifiers for this boundary condition
    normal = True


def registered_boundary_condition_classes() -> dict[str, type[BCBase]]:
    """returns all boundary condition classes that are currently defined

    Returns:
        dict: a dictionary with the names of the boundary condition classes
    """
    return {
        cls_name: cls
        for cls_name, cls in BCBase._subclasses.items()
        if not ("Base" in cls_name or cls_name.startswith("_"))  # skip internal classes
    }


def registered_boundary_condition_names() -> dict[str, type[BCBase]]:
    """returns all named boundary conditions that are currently defined

    Returns:
        dict: a dictionary with the names of the boundary conditions that can be used
    """
    return {cls_name: cls for cls_name, cls in BCBase._conditions.items()}
