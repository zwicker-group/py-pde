"""Defines the base class for all grids.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import json
import logging
import math
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..tools.cache import cached_method, cached_property
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import hybridmethod
from .coordinates import CoordinatesBase, DimensionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray

    from ..backends.base import BackendBase, OperatorInfo
    from ..tools.typing import (
        CellVolume,
        FloatingArray,
        FloatOrArray,
        Number,
        NumberOrArray,
        NumericArray,
        OperatorImplType,
        OperatorType,
    )
    from ._mesh import GridMesh
    from .boundaries.axes import BoundariesBase, BoundariesData

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for grids."""

PI_4 = 4 * np.pi
PI_43 = 4 / 3 * np.pi
CoordsType = Literal["cartesian", "grid", "cell"]


def _check_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    """Checks the consistency of shape tuples."""
    if hasattr(shape, "__iter__"):
        shape_list: Sequence[int] = shape  # type: ignore
    else:
        shape_list = [shape]

    if len(shape_list) == 0:
        msg = "Require at least one dimension"
        raise ValueError(msg)

    # convert the shape to a tuple of integers
    result = []
    for dim in shape_list:
        if dim == int(dim) and dim >= 1:
            result.append(int(dim))
        else:
            msg = f"{dim!r} is not a valid number of support points"
            raise ValueError(msg)
    return tuple(result)


def discretize_interval(
    x_min: float, x_max: float, num: int
) -> tuple[FloatingArray, float]:
    r"""Construct a list of equidistantly placed intervals.

    The discretization is defined as

    .. math::
            x_i &= x_\mathrm{min} + \left(i + \frac12\right) \Delta x
            \quad \text{for} \quad i = 0, \ldots, N - 1
        \\
            \Delta x &= \frac{x_\mathrm{max} - x_\mathrm{min}}{N}

    where :math:`N` is the number of intervals given by `num`.

    Args:
        x_min (float): Minimal value of the axis
        x_max (float): Maximal value of the axis
        num (int): Number of intervals

    Returns:
        tuple: (midpoints, dx): the midpoints of the intervals and the used
        discretization `dx`.
    """
    dx = (x_max - x_min) / num
    return (np.arange(num) + 0.5) * dx + x_min, dx


class DomainError(ValueError):
    """Exception indicating that point lies outside domain."""


class PeriodicityError(RuntimeError):
    """Exception indicating that the grid periodicity is inconsistent."""


class GridBase(metaclass=ABCMeta):
    """Base class for all grids defining common methods and interfaces."""

    # class properties
    _subclasses: dict[str, type[GridBase]] = {}  # all classes inheriting from this
    _logger: logging.Logger  # logger instance to output information

    # properties that are defined in subclasses
    c: CoordinatesBase
    """:class:`~pde.grids.coordinates.CoordinatesBase`: Coordinates of the grid."""
    axes: list[str]
    """list: Names of all axes that are described by the grid"""
    axes_symmetric: list[str] = []
    """list: The names of the additional axes that the fields do not depend on,
    e.g. along which they are constant. """

    boundary_names: dict[str, tuple[int, bool]] = {}
    """dict: Names of boundaries to select them conveniently"""
    cell_volume_data: Sequence[FloatOrArray] | None
    """list: Information about the size of discretization cells"""
    coordinate_constraints: list[int] = []
    """list: axes that not described explicitly"""
    num_axes: int
    """int: Number of axes that are *not* assumed symmetrically"""

    # mandatory, immutable, private attributes
    _axes_symmetric: tuple[int, ...] = ()
    _axes_described: tuple[int, ...]
    _axes_bounds: tuple[tuple[float, float], ...]
    _axes_coords: tuple[FloatingArray, ...]
    _discretization: FloatingArray
    _periodic: list[bool]
    _shape: tuple[int, ...]

    # to help sphinx, we here list docstrings for classproperties
    operators: set[str]
    """ set: names of all operators defined for this grid """

    def __init__(self) -> None:
        """Initialize the grid."""
        self._mesh: GridMesh | None = None

        self._axes_described = tuple(
            i for i in range(self.dim) if i not in self._axes_symmetric
        )
        self.num_axes = len(self._axes_described)
        self.axes = [self.c.axes[i] for i in self._axes_described]
        self.axes_symmetric = [self.c.axes[i] for i in self._axes_symmetric]

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)

        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

        # register all subclasses to reconstruct them later
        if cls is not GridBase:
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}", stacklevel=2)
            cls._subclasses[cls.__name__] = cls

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_cache_methods", None)  # delete method cache if present
        return state

    @classmethod
    def from_state(cls, state: str | dict[str, Any]) -> GridBase:
        """Create a field from a stored `state`.

        Args:
            state (`str` or `dict`):
                The state from which the grid is reconstructed. If `state` is a
                string, it is decoded as JSON, which should yield a `dict`.

        Returns:
            :class:`GridBase`: Grid re-created from the state
        """
        # decode the json data
        if isinstance(state, str):
            state = dict(json.loads(state))

        # create the instance of the correct class
        class_name = state.pop("class")
        if class_name == cls.__name__:
            msg = f"Cannot reconstruct abstract class `{class_name}`"
            raise RuntimeError(msg)
        grid_cls = cls._subclasses[class_name]
        return grid_cls.from_state(state)

    @classmethod
    def from_bounds(
        cls,
        bounds: Sequence[tuple[float, float]],
        shape: Sequence[int],
        periodic: Sequence[bool],
    ) -> GridBase:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """int: The spatial dimension in which the grid is embedded"""
        return self.c.dim

    @property
    def periodic(self) -> list[bool]:
        """list: Flags that describe which axes are periodic"""
        return self._periodic

    @property
    def axes_bounds(self) -> tuple[tuple[float, float], ...]:
        """tuple: lower and upper bounds of each axis"""
        return self._axes_bounds

    @property
    def axes_coords(self) -> tuple[FloatingArray, ...]:
        """tuple: coordinates of the cells for each axis"""
        return self._axes_coords

    def get_axis_index(self, key: int | str, allow_symmetric: bool = True) -> int:
        """Return the index belonging to an axis.

        Args:
            key (int or str):
                The index or name of an axis
            allow_symmetric (bool):
                Whether axes with assumed symmetry are included

        Returns:
            int: The index of the axis
        """
        if isinstance(key, str):
            # determine key index from name of the axis
            if allow_symmetric:
                axes = self.axes + self.axes_symmetric
            else:
                axes = self.axes
            if key in axes:
                return axes.index(key)
            msg = f"`{key}` is not in the axes {axes}"
            raise IndexError(msg)
        if isinstance(key, int):
            # assume that it is already an index
            return key
        msg = "Index must be an integer or the name of an axes"
        raise IndexError(msg)

    def _get_boundary_index(self, index: str | tuple[int, bool]) -> tuple[int, bool]:
        """Return the index of a boundary belonging to an axis.

        Args:
            index (str or tuple):
                Index specifying the boundary. Can be either a string given in
                :attr:`~pde.grids.base.GridBase.boundary_names`, like :code:`"left"`, or
                a tuple of the axis index perpendicular to the boundary and a boolean
                specifying whether the boundary is at the upper side of the axis or not,
                e.g., :code:`(1, True)`.

        Returns:
            tuple: axis index perpendicular to the boundary and a boolean specifying
                whether the boundary is at the upper side of the axis or not.
        """
        if isinstance(index, str):
            # assume that the index is a known identifier
            if index in self.boundary_names:
                # found a known boundary
                axis, upper = self.boundary_names[index]
            else:
                # check all axes
                for axis, ax_name in enumerate(self.axes):
                    if index == ax_name + "-":
                        upper = False
                        break
                    if index == ax_name + "+":
                        upper = True
                        break
                else:
                    msg = "Unknown boundary {index}"
                    raise KeyError(msg)

        else:
            # assume the index is directly given as a tuple of an axis and a boolean
            axis, upper = index
        return axis, upper

    @property
    def discretization(self) -> FloatingArray:
        """:class:`numpy.array`: the linear size of a cell along each axis."""
        return self._discretization

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int: the number of support points of each axis"""
        return self._shape

    @property
    def num_cells(self) -> int:
        """int: the number of cells in this grid"""
        return math.prod(self.shape)

    @property
    def _shape_full(self) -> tuple[int, ...]:
        """tuple of int: number of support points including ghost points"""
        return tuple(num + 2 for num in self.shape)

    @property
    def _idx_valid(self) -> tuple[slice, ...]:
        """tuple: slices to extract valid data from full data"""
        return tuple(slice(1, s + 1) for s in self.shape)

    def _make_get_valid(self) -> Callable[[NumericArray], NumericArray]:
        """Create a function to extract the valid part of a full data array.

        Returns:
            callable: Mapping a numpy array containing the full data of the grid to a
                numpy array of only the valid data
        """
        # deprecated on 2025-12-06
        from ..backends.numba.utils import make_get_valid

        warnings.warn(
            "`_make_get_valid` is deprecated. Use "
            "`pde.backends.numba.utils.make_get_valid` instead.",
            stacklevel=2,
        )
        return make_get_valid(self)

    @overload
    def _make_set_valid(self) -> Callable[[NumericArray, NumericArray], None]: ...

    @overload
    def _make_set_valid(
        self, bcs: BoundariesBase
    ) -> Callable[[NumericArray, NumericArray, dict], None]: ...

    def _make_set_valid(
        self,
        bcs: BoundariesBase | None = None,
        *,
        backend: str | BackendBase = "config",
    ) -> Callable:
        """Create a function to set the valid part of a full data array.

        Args:
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                If supplied, the returned function also enforces boundary conditions by
                setting the ghost cells to the correct values
            backend (str):
                The backend to use for making the operator

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        from ..backends import backends

        return backends[backend].make_data_setter(self, bcs=bcs)

    @property
    @abstractmethod
    def state(self) -> dict[str, Any]:
        """dict: all information required for reconstructing the grid"""

    @property
    def state_serialized(self) -> str:
        """str: JSON-serialized version of the state of this grid"""
        state = self.state
        state["class"] = self.__class__.__name__
        return json.dumps(state)

    def copy(self) -> GridBase:
        """Return a copy of the grid."""
        return self.__class__.from_state(self.state)

    __copy__ = copy

    def __deepcopy__(self, memo: dict[int, Any]) -> GridBase:
        """Create a deep copy of the grid.

        This function is for instance called when
        a grid instance appears in another object that is copied using `copy.deepcopy`
        """
        # this implementation assumes that a simple call to copy is sufficient
        result = self.copy()
        memo[id(self)] = result
        return result

    def __repr__(self) -> str:
        """Return instance as string."""
        args = ", ".join(str(k) + "=" + str(v) for k, v in self.state.items())
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.axes_bounds == other.axes_bounds
            and self.periodic == other.periodic
        )

    def _cache_hash(self) -> int:
        """Returns a value to determine when a cache needs to be updated."""
        return hash(
            (
                self.__class__.__name__,
                self.shape,
                self.axes_bounds,
                tuple(self.periodic),
            )
        )

    def compatible_with(self, other: GridBase) -> bool:
        """Tests whether this grid is compatible with other grids.

        Grids are compatible when they cover the same area with the same
        discretization. The difference to equality is that compatible grids do
        not need to have the same periodicity in their boundaries.

        Args:
            other (:class:`~pde.grids.base.GridBase`):
                The other grid to test against

        Returns:
            bool: Whether the grid is compatible
        """
        return (
            self.__class__ == other.__class__
            and self.shape == other.shape
            and self.axes_bounds == other.axes_bounds
        )

    def assert_grid_compatible(self, other: GridBase) -> None:
        """Checks whether `other` is compatible with the current grid.

        Args:
            other (:class:`~pde.grids.base.GridBase`):
                The grid compared to this one

        Raises:
            ValueError: if grids are not compatible
        """
        if not self.compatible_with(other):
            msg = f"Grids {self} and {other} are incompatible"
            raise ValueError(msg)

    @property
    def numba_type(self) -> str:
        """str: represents type of the grid data in numba signatures"""
        # deprecated since 2025-11-19
        warnings.warn(
            "`numba_type` property is deprecated. Use the method `get_grid_numba_type` "
            "in the numba backend instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..backends.numba.grids import get_grid_numba_type

        return get_grid_numba_type(self)

    @cached_property()
    def coordinate_arrays(self) -> tuple[FloatingArray, ...]:
        """tuple: for each axes: coordinate values for all cells"""
        return tuple(np.meshgrid(*self.axes_coords, indexing="ij"))

    @cached_property()
    def cell_coords(self) -> FloatingArray:
        """:class:`~numpy.ndarray`: coordinate values for all axes of each cell."""
        return np.moveaxis(self.coordinate_arrays, 0, -1)

    @cached_property()
    def cell_volumes(self) -> FloatingArray:
        """:class:`~numpy.ndarray`: volume of each cell."""
        if self.cell_volume_data is None:
            # use the self.c to calculate cell volumes
            d2 = self.discretization / 2
            x_low = self._coords_full(self.cell_coords - d2, value="min")
            x_high = self._coords_full(self.cell_coords + d2, value="max")
            return self.c.cell_volume(x_low, x_high)

        # use cell_volume_data
        vols = functools.reduce(np.outer, self.cell_volume_data)
        return np.broadcast_to(vols, self.shape)

    @cached_property()
    def uniform_cell_volumes(self) -> bool:
        """bool: returns True if all cell volumes are the same"""
        if self.cell_volume_data is None:
            return False
        return all(np.asarray(vols).ndim == 0 for vols in self.cell_volume_data)

    def _difference_vector(
        self,
        p1: FloatingArray,
        p2: FloatingArray,
        *,
        coords: CoordsType,
        periodic: Sequence[bool],
        axes_bounds: tuple[tuple[float, float], ...] | None,
    ) -> FloatingArray:
        """Return Cartesian vector(s) pointing from p1 to p2.

        In case of periodic boundary conditions, the shortest vector is returned.

        Args:
            p1 (:class:`~numpy.ndarray`):
                First point(s)
            p2 (:class:`~numpy.ndarray`):
                Second point(s)
            coords (str):
                The coordinate system in which the points are specified.
            periodic (sequence of bool):
                Indicates which cartesian axes are periodic
            axes_bounds (sequence of pair of floats):
                Indicates the bounds of the cartesian axes

        Returns:
            :class:`~numpy.ndarray`: The difference vectors between the points with
            periodic boundary conditions applied.
        """
        x1 = self.transform(p1, source=coords, target="cartesian")
        x2 = self.transform(p2, source=coords, target="cartesian")
        if axes_bounds is None:
            axes_bounds = self.axes_bounds

        diff = np.atleast_1d(x2) - np.atleast_1d(x1)
        assert diff.shape[-1] == self.dim

        for i, per in enumerate(periodic):
            if per:
                size = axes_bounds[i][1] - axes_bounds[i][0]
                diff[..., i] = (diff[..., i] + size / 2) % size - size / 2
        return diff

    def difference_vector(
        self, p1: FloatingArray, p2: FloatingArray, *, coords: CoordsType = "grid"
    ) -> FloatingArray:
        """Return Cartesian vector(s) pointing from p1 to p2.

        In case of periodic boundary conditions, the shortest vector is returned.

        Args:
            p1 (:class:`~numpy.ndarray`):
                First point(s)
            p2 (:class:`~numpy.ndarray`):
                Second point(s)
            coords (str):
                Coordinate system in which points are specified. Valid values are
                `cartesian`, `cell`, and `grid`;
                see :meth:`~pde.grids.base.GridBase.transform`.

        Returns:
            :class:`~numpy.ndarray`: The difference vectors between the points with
            periodic boundary conditions applied.
        """
        return self._difference_vector(
            p1, p2, coords=coords, periodic=[False] * self.dim, axes_bounds=None
        )

    def distance(
        self, p1: FloatingArray, p2: FloatingArray, *, coords: CoordsType = "grid"
    ) -> float:
        """Calculate the distance between two points given in real coordinates.

        This takes periodic boundary conditions into account if necessary.

        Args:
            p1 (:class:`~numpy.ndarray`):
                First position
            p2 (:class:`~numpy.ndarray`):
                Second position
            coords (str):
                Coordinate system in which points are specified. Valid values are
                `cartesian`, `cell`, and `grid`;
                see :meth:`~pde.grids.base.GridBase.transform`.

        Returns:
            float: Distance between the two positions
        """
        diff = self.difference_vector(p1, p2, coords=coords)
        return np.linalg.norm(diff, axis=-1)  # type: ignore

    def _iter_boundaries(self) -> Iterator[tuple[int, bool]]:
        """Iterate over all boundaries of the grid.

        Yields:
            tuple: for each boundary, the generator returns a tuple indicating
            the axis of the boundary together with a boolean value indicating
            whether the boundary lies on the upper side of the axis.
        """
        return itertools.product(range(self.num_axes), [True, False])

    def _boundary_coordinates(
        self, axis: int, upper: bool, *, offset: float = 0
    ) -> FloatingArray:
        """Get coordinates of points on the boundary.

        Args:
            axis (int):
                The axis perpendicular to the boundary
            upper (bool):
                Whether the boundary is at the upper side of the axis
            offset (float):
                A distance by which the points will be moved away from the boundary.
                Positive values move the points into the interior of the domain

        Returns:
            :class:`~numpy.ndarray`: Coordinates of the boundary points. This array has
            one less dimension than the grid has axes.
        """
        # get coordinate along the axis determining the boundary
        if upper:
            c_bndry = np.array([self._axes_bounds[axis][1]]) - offset
        else:
            c_bndry = np.array([self._axes_bounds[axis][0]]) + offset

        # get orthogonal coordinates
        coords = tuple(
            c_bndry if i == axis else self._axes_coords[i] for i in range(self.num_axes)
        )
        points = np.meshgrid(*coords, indexing="ij")

        # assemble into array
        shape_bndry = tuple(self.shape[i] for i in range(self.num_axes) if i != axis)
        shape = (*shape_bndry, self.num_axes)
        return np.stack(points, -1).reshape(shape)

    @property
    def volume(self) -> float:
        """float: total volume of the grid"""
        # this property should be overwritten when the volume can be calculated directly
        return self.cell_volumes.sum()  # type: ignore

    def point_to_cartesian(self, points: FloatingArray) -> FloatingArray:
        """Convert coordinates of a point in grid coordinates to Cartesian coordinates.

        Args:
            points (:class:`~numpy.ndarray`):
                The grid coordinates of the points

        Returns:
            :class:`~numpy.ndarray`: The Cartesian coordinates of the point
        """
        return self.c.pos_to_cart(self._coords_full(points))

    def point_from_cartesian(self, points: FloatingArray) -> FloatingArray:
        """Convert points given in Cartesian coordinates to grid coordinates.

        Args:
            points (:class:`~numpy.ndarray`):
                Points given in Cartesian coordinates.

        Returns:
            :class:`~numpy.ndarray`: Points given in the coordinates of the grid
        """
        return self._coords_symmetric(self.c.pos_from_cart(points))

    def _vector_to_cartesian(
        self, points: FloatingArray, components: ArrayLike
    ) -> NumericArray:
        """Convert the vectors at given points into a Cartesian basis.

        Args:
            points (:class:`~numpy.ndarray`):
                The coordinates of the point(s) where the vectors are specified. These
                need to be given in grid coordinates.
            components (:class:`~numpy.ndarray`):
                The components of the vectors at the given points

        Returns:
            The vectors specified at the same position but with components given in
            Cartesian coordinates
        """
        points = np.asanyarray(points)
        components = np.asanyarray(components)
        # check input shapes
        if points.shape[-1] != self.dim:
            msg = f"`points` must have {self.dim} coordinates"
            raise DimensionError(msg)
        shape = points.shape[:-1]  # shape of array describing the different points
        vec_shape = (self.dim, *shape)
        if components.shape != vec_shape:
            msg = f"`components` must have shape {vec_shape}"
            raise DimensionError(msg)

        # convert the basis of the vectors to Cartesian
        rot_mat = self.c.basis_rotation(points)
        assert rot_mat.shape == (self.dim, self.dim) or rot_mat.shape == (
            self.dim,
            self.dim,
            *shape,
        )
        return np.einsum("j...,ji...->i...", components, rot_mat)  # type: ignore

    def normalize_point(
        self, point: FloatingArray, *, reflect: bool = False
    ) -> FloatingArray:
        """Normalize grid coordinates by applying periodic boundary conditions.

        Here, points are assumed to be specified by the physical values along the
        non-symmetric axes of the grid, e.g., by grid coordinates. Normalizing points is
        useful to make sure they lie within the domain of the  grid. This function
        respects periodic boundary conditions and can also reflect points off the
        boundary if `reflect = True`.

        Args:
            point (:class:`~numpy.ndarray`):
                Coordinates of a single point or an array of points, where the last axis
                denotes the point coordinates (e.g., a list of points).
            reflect (bool):
                Flag determining whether coordinates along non-periodic axes are
                reflected to lie in the valid range. If `False`, such coordinates are
                left unchanged and only periodic boundary conditions are enforced.

        Returns:
            :class:`~numpy.ndarray`: The respective coordinates with periodic
            boundary conditions applied.
        """
        point = np.asarray(point, dtype=np.double)
        if point.size == 0:
            return np.zeros((0, self.num_axes))

        if point.ndim == 0:
            if self.num_axes > 1:
                msg = f"Point {point} is not of dimension {self.num_axes}"
                raise DimensionError(msg)
        elif point.shape[-1] != self.num_axes:
            msg = (
                f"Array of shape {point.shape} does not describe points of dimension "
                f"{self.num_axes}"
            )
            raise DimensionError(msg)

        # normalize the coordinates for the periodic dimensions
        bounds = np.array(self.axes_bounds)
        xmin = bounds[:, 0]
        xmax = bounds[:, 1]
        xdim = xmax - xmin

        if self.num_axes == 1:
            # single dimension
            if self.periodic[0]:
                point = (point - xmin[0]) % xdim[0] + xmin[0]
            elif reflect:
                arg = (point - xmax[0]) % (2 * xdim[0]) - xdim[0]
                point = xmin[0] + np.abs(arg)

        else:
            # multiple dimensions
            for i in range(self.num_axes):
                if self.periodic[i]:
                    point[..., i] = (point[..., i] - xmin[i]) % xdim[i] + xmin[i]
                elif reflect:
                    arg = (point[..., i] - xmax[i]) % (2 * xdim[i]) - xdim[i]
                    point[..., i] = xmin[i] + np.abs(arg)

        return point

    def _coords_symmetric(self, points: FloatingArray) -> FloatingArray:
        """Return only non-symmetric point coordinates.

        Args:
            points (:class:`~numpy.ndarray`):
                The points specified with `dim` coordinates

        Returns:
            :class:`~numpy.ndarray`: The points with only `num_axes` coordinates, which
            are not along symmetry axes of the grid.
        """
        if points.shape[-1] != self.dim:
            msg = f"Points need to be specified as {self.c.axes}"
            raise DimensionError(msg)
        return points[..., self._axes_described]

    def _coords_full(
        self, points: FloatingArray, *, value: Literal["min", "max"] | float = 0.0
    ) -> FloatingArray:
        """Specify point coordinates along symmetric axes on grids.

        Args:
            points (:class:`~numpy.ndarray`):
                The points specified with `num_axes` coordinates, not specifying
                coordinates along symmetry axes of the grid.
            value (str or float):
                Value of the points along symmetry axes. The special values `min` and
                `max` denote the minimal and maximal values along the respective
                coordinates.

        Returns:
            :class:`~numpy.ndarray`: The points with all `dim` coordinates
        """
        if self.num_axes == self.dim:
            return points
        if points.shape[-1] != self.num_axes:
            msg = f"Points need to be specified as {self.axes}"
            raise DimensionError(msg)
        res = np.empty((*points.shape[:-1], self.dim), dtype=points.dtype)
        j = 0
        for i in range(self.dim):
            if i in self._axes_described:
                res[..., i] = points[..., j]
                j += 1
            else:
                if value == "min":
                    res[..., i] = self.c.coordinate_limits[i][0]
                elif value == "max":
                    res[..., i] = self.c.coordinate_limits[i][1]
                else:
                    res[..., i] = value
        return res

    def transform(
        self, coordinates: FloatingArray, source: CoordsType, target: CoordsType
    ) -> FloatingArray:
        """Converts coordinates from one coordinate system to another.

        Supported coordinate systems include the following:

        * `cartesian`:
            Cartesian coordinates where each point carries `dim` values. These are the
            true physical coordinates in space.
        * `grid`:
            Coordinates values in the coordinate system defined by the grid. A point is
            thus characterized by `grid.num_axes` values.
        * `cell`:
            Normalized grid coordinates based on indexing the discretization cells. A
            point is characterized by `grid.num_axes` values and the range of values for
            a given axis is between `0` and `N`, where `N` is the number of grid points.
            Consequently, the integral part of the cell coordinate denotes the cell,
            while the fractional part denotes the relative position within the cell. In
            particular, the cell center is located at `i + 0.5` with `i = 0, ..., N-1`.

        Note:
            Some conversion might involve projections if the coordinate system imposes
            symmetries. For instance, converting 3d Cartesian coordinates to grid
            coordinates in a spherically symmetric  grid will only return the radius
            from the origin. Conversely, converting these grid coordinates back to 3d
            Cartesian coordinates will only return coordinates along a particular ray
            originating at the origin.

        Args:
            coordinates (:class:`~numpy.ndarray`):
                The coordinates to convert
            source (str):
                The source coordinate system
            target (str):
                The target coordinate system

        Returns:
            :class:`~numpy.ndarray`: The transformed coordinates
        """
        if source == "cartesian":
            # Cartesian coordinates given
            cartesian = np.atleast_1d(coordinates)
            if cartesian.shape[-1] != self.dim:
                msg = f"Require {self.dim} cartesian coordinates"
                raise DimensionError(msg)

            if target == "cartesian":
                return coordinates

            # convert Cartesian coordinates to grid coordinates
            grid_coords = self.point_from_cartesian(cartesian)

            if target == "grid":
                return grid_coords
            if target == "cell":
                c_min = np.array(self.axes_bounds)[:, 0]
                return (grid_coords - c_min) / self.discretization  # type: ignore

        elif source == "cell":
            # Cell coordinates given
            cells = np.atleast_1d(coordinates)
            if cells.shape[-1] != self.num_axes:
                msg = f"Require {self.num_axes} cell coordinates"
                raise DimensionError(msg)

            if target == "cell":
                return coordinates

            # convert cell coordinates to grid coordinates
            c_min = np.array(self.axes_bounds)[:, 0]
            grid_coords = c_min + cells * self.discretization

            if target == "grid":
                return grid_coords
            if target == "cartesian":
                return self.point_to_cartesian(grid_coords)

        elif source == "grid":
            # Grid coordinates given
            grid_coords = np.atleast_1d(coordinates)
            if grid_coords.shape[-1] != self.num_axes:
                msg = f"Require {self.num_axes} grid coordinates"
                raise DimensionError(msg)

            if target == "cartesian":
                return self.point_to_cartesian(grid_coords)
            if target == "cell":
                c_min = np.array(self.axes_bounds)[:, 0]
                return (grid_coords - c_min) / self.discretization  # type: ignore
            if target == "grid":
                return grid_coords

        else:
            msg = f"Unknown source coordinates `{source}`"
            raise ValueError(msg)
        msg = f"Unknown target coordinates `{target}`"
        raise ValueError(msg)

    def contains_point(
        self,
        points: FloatingArray,
        *,
        coords: Literal["cartesian", "cell", "grid"] = "cartesian",
    ) -> NDArray[np.bool]:
        """Check whether the point is contained in the grid.

        Args:
            points (:class:`~numpy.ndarray`):
                Coordinates of the point(s)
            coords (str):
                The coordinate system in which the points are given

        Returns:
            :class:`~numpy.ndarray`: A boolean array indicating which points lie within
            the grid
        """
        cell_coords = self.transform(points, source=coords, target="cell")
        return np.all((cell_coords >= 0) & (cell_coords <= self.shape), axis=-1)  # type: ignore

    def iter_mirror_points(
        self, point: FloatingArray, with_self: bool = False, only_periodic: bool = True
    ) -> Iterator[FloatingArray]:
        """Generates all mirror points corresponding to `point`

        Args:
            point (:class:`~numpy.ndarray`):
                The point within the grid
            with_self (bool):
                Whether to include the point itself
            only_periodic (bool):
                Whether to only mirror along periodic axes

        Returns:
            A generator yielding the coordinates that correspond to mirrors
        """
        # the default implementation does not know about mirror points
        if with_self:
            yield np.asanyarray(point, dtype=np.double)

    @fill_in_docstring
    def get_boundary_conditions(
        self, bc: BoundariesData = "auto_periodic_neumann", rank: int = 0
    ) -> BoundariesBase:
        """Constructs boundary conditions from a flexible data format.

        Args:
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            rank (int):
                The tensorial rank of the value associated with the boundary conditions.

        Returns:
            :class:`~pde.grids.boundaries.axes.BoundariesBase`: The boundary conditions
            for all axes.

        Raises:
            ValueError:
                If the data given in `bc` cannot be read
            PeriodicityError:
                If the boundaries are not compatible with the periodic axes of the grid.
        """
        from .boundaries import BoundariesBase

        if self._mesh is None:
            # get boundary conditions for a simple grid that is not part of a mesh
            bcs = BoundariesBase.from_data(bc, grid=self, rank=rank)

        else:
            # this grid is part of a mesh and we thus need to set special conditions to
            # support parallelism via MPI. We here assume that bc is given for the full
            # system and not
            bcs_base = BoundariesBase.from_data(bc, grid=self._mesh.basegrid, rank=rank)
            bcs = self._mesh.extract_boundary_conditions(bcs_base)

        return bcs

    def get_line_data(
        self, data: NumericArray, extract: str = "auto"
    ) -> dict[str, Any]:
        """Return a line cut through the grid.

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points
            extract (str):
                Determines which cut is done through the grid. Possible choices depend
                on the actual grid.

        Returns:
            dict: A dictionary with information about the line cut, which is convenient
            for plotting.
        """
        raise NotImplementedError

    def get_image_data(self, data: NumericArray) -> dict[str, Any]:
        """Return a 2d-image of the data.

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points

        Returns:
            dict: A dictionary with information about the data convenient for plotting.
        """
        raise NotImplementedError

    def get_vector_data(self, data: NumericArray, **kwargs) -> dict[str, Any]:
        r"""Return data to visualize vector field.

        Args:
            data (:class:`~numpy.ndarray`):
                The vectorial values at the grid points
            \**kwargs:
                Arguments forwarded to
                :meth:`~pde.grids.base.GridBase.get_image_data`.

        Returns:
            dict: A dictionary with information about the data convenient for plotting.
        """
        if self.dim != 2:
            msg = "Can only plot generic vector fields for dim=2"
            raise DimensionError(msg)
        if data.shape != (self.dim, *self.shape):
            msg = (
                f"Shape {data.shape} of the data array is not compatible with grid "
                f"shape {self.shape}"
            )
            raise ValueError(msg)

        # obtain the correctly interpolated components of the vector in grid coordinates
        img_coord0 = self.get_image_data(data[0], **kwargs)
        img_coord1 = self.get_image_data(data[1], **kwargs)

        points_cart = np.stack((img_coord0["xs"], img_coord0["ys"]), axis=-1)
        points = self.c._pos_from_cart(points_cart)

        # convert vectors to cartesian coordinates
        img_data = img_coord0
        img_data["data_x"], img_data["data_y"] = self._vector_to_cartesian(
            points, [img_coord0["data"], img_coord1["data"]]
        )
        img_data.pop("data")
        return img_data

    def get_random_point(
        self,
        *,
        boundary_distance: float = 0,
        coords: CoordsType = "cartesian",
        rng: np.random.Generator | None = None,
    ) -> FloatingArray:
        """Return a random point within the grid.

        Args:
            boundary_distance (float):
                The minimal distance this point needs to have from all boundaries.
            coords (str):
                Determines the coordinate system in which the point is specified. Valid
                values are `cartesian`, `cell`, and `grid`;
                see :meth:`~pde.grids.base.GridBase.transform`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)

        Returns:
            :class:`~numpy.ndarray`: The coordinates of the random point
        """
        raise NotImplementedError

    @hybridmethod  # type: ignore
    @property
    def operators(cls) -> set[str]:
        """set: all operators defined for this class"""
        from ..backends import backends

        # get all operators registered on the class
        operators = set()
        for name in backends:
            with contextlib.suppress(ImportError):
                operators |= backends[name].get_registered_operators(cls)
        return operators

    @operators.instancemethod  # type: ignore
    @property
    def operators(self) -> set[str]:
        """set: all operators defined for this instance"""
        from ..backends import backends

        # get all operators registered on the class
        operators = set()
        for name in backends:
            with contextlib.suppress(ImportError):
                operators |= backends[name].get_registered_operators(self)
        return operators

    @cached_method()
    def make_operator_no_bc(
        self,
        operator: str | OperatorInfo,
        *,
        backend: str | BackendBase = "config",
        **kwargs,
    ) -> OperatorImplType:
        """Return a compiled function applying an operator without boundary conditions.

        A function that takes the discretized full data as an input and an array of
        valid data points to which the result of applying the operator is written.

        Note:
            The resulting function does not check whether the ghost cells of the input
            array have been supplied with sensible values. It is the responsibility of
            the user to set the values of the ghost cells beforehand. Use this function
            only if you absolutely know what you're doing. In all other cases,
            :meth:`make_operator` is probably the better choice.

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            backend (str):
                The backend to use for making the operator
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray), so they `out` array need
            to be supplied explicitly.
        """
        from ..backends import backends

        # determine the operator for the chosen backend
        operator_info = backends[backend].get_operator_info(self, operator)
        return operator_info.factory(self, **kwargs)

    @cached_method()
    @fill_in_docstring
    def make_operator(
        self,
        operator: str | OperatorInfo,
        bc: BoundariesData,
        *,
        backend: str | BackendBase = "config",
        **kwargs,
    ) -> OperatorType:
        """Return a compiled function applying an operator with boundary conditions.

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            backend (str):
                The backend to use for making the operator
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array.

        The function also accepts an optional parameter `args`, which is forwarded to
        `set_ghost_cells`. This allows setting boundary conditions based on external
        parameters, like time. Note that since the returned operator will always be
        compiled by Numba, the arguments need to be compatible with Numba. The
        following example shows how to pass the current time `t`:

        .. code-block:: python

            from pde.backends.numba.utils import numba_dict

            operator = grid.make_operator("laplace", bc={"value_expression": "t"})
            operator(field.data, args=numba_dict(t=t))

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        from ..backends import backends

        # determine the operator for the chosen backend
        backend_impl = backends[backend]
        operator_info = backend_impl.get_operator_info(self, operator)

        # set the boundary conditions before applying this operator
        bcs = self.get_boundary_conditions(bc, rank=operator_info.rank_in)
        return backend_impl.make_operator(self, operator_info, bcs=bcs)

    def slice(self, indices: Sequence[int]) -> GridBase:
        """Return a subgrid of only the specified axes.

        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid

        Returns:
            :class:`GridBase`: The subgrid
        """
        msg = f"Slicing is not implemented for class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def plot(self) -> None:
        """Visualize the grid."""
        msg = f"Plotting is not implemented for class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    @property
    def typical_discretization(self) -> float:
        """float: the average side length of the cells"""
        return np.mean(self.discretization)  # type: ignore

    def integrate(
        self, data: NumberOrArray, axes: int | Sequence[int] | None = None
    ) -> NumberOrArray:
        """Integrates the discretized data over the grid.

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the support points of the grid that need to be
                integrated.
            axes (list of int, optional):
                The axes along which the integral is performed. If omitted, all
                axes are integrated over.

        Returns:
            :class:`~numpy.ndarray`: The values integrated over the entire grid
        """
        # determine the volumes of the individual cells
        if self.cell_volume_data is None:
            if axes is None:
                cell_volumes = self.cell_volumes
            else:
                raise NotImplementedError
        else:
            if axes is None:
                volume_list = self.cell_volume_data
            else:
                # use stored value for the default case of integrating over all axes
                if isinstance(axes, int):
                    axes = (axes,)
                else:
                    axes = tuple(axes)  # required for numpy.sum
                volume_list = [
                    cell_vol if ax in axes else 1
                    for ax, cell_vol in enumerate(self.cell_volume_data)
                ]
            cell_volumes = functools.reduce(np.outer, volume_list)

        # determine the axes over which we will integrate
        if not isinstance(data, np.ndarray) or data.ndim < self.num_axes:
            # deal with the case where data is not supplied for each support
            # point, e.g., when a single scalar is integrated over the grid
            data = np.broadcast_to(data, self.shape)

        elif data.ndim > self.num_axes:
            # deal with the case where more than a single value is provided per
            # support point, e.g., when a tensorial field is integrated
            offset = data.ndim - self.num_axes
            if axes is None:
                # integrate over all axes of the grid
                axes = tuple(range(offset, data.ndim))
            else:
                # shift the indices to account for the data shape
                axes = tuple(offset + i for i in axes)

        # calculate integral using a weighted sum along the chosen axes
        integral = (data * cell_volumes).sum(axis=axes)

        if self._mesh is None or len(self._mesh) == 1:
            # standard case of a single integral
            return integral  # type: ignore

        # we are in a parallel run, so we need to gather the sub-integrals from all
        from mpi4py.MPI import COMM_WORLD

        integral_full = np.empty_like(integral)
        COMM_WORLD.Allreduce(integral, integral_full)
        return integral_full  # type: ignore

    def make_normalize_point_compiled(
        self, reflect: bool = True
    ) -> Callable[[FloatingArray], None]:
        """Return a compiled function that normalizes a point.

        Here, the point is assumed to be specified by the physical values along
        the non-symmetric axes of the grid. Normalizing points is useful to make sure
        they lie within the domain of the  grid. This function respects periodic
        boundary conditions and can also reflect points off the boundary.

        Args:
            reflect (bool):
                Flag determining whether coordinates along non-periodic axes are
                reflected to lie in the valid range. If `False`, such coordinates are
                left unchanged and only periodic boundary conditions are enforced.

        Returns:
            callable: A function that takes a :class:`~numpy.ndarray` as an argument,
            which describes the coordinates of the points. This array is modified
            in-place!
        """
        # Removed on 2025-12-06
        msg = "This method is no longer supported"
        raise NotImplementedError(msg)

    @cached_method()
    def make_cell_volume_compiled(self, flat_index: bool = False) -> CellVolume:
        """Return a compiled function returning the volume of a grid cell.

        Args:
            flat_index (bool):
                When True, cell_volumes are indexed by a single integer into the
                flattened array.

        Returns:
            function: returning the volume of the chosen cell
        """
        from ..backends.numba.grids import make_cell_volume_getter

        # deprecated since 2025-11-19
        warnings.warn(
            "`make_cell_volume_compiled` is deprecated. Use "
            "`pde.backends.numba.grids.make_cell_volume_getter` instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return make_cell_volume_getter(grid=self, flat_index=flat_index)

    @cached_method()
    def _make_interpolator_compiled(
        self,
        *,
        fill: Number | None = None,
        with_ghost_cells: bool = False,
        cell_coords: bool = False,
    ) -> Callable[[NumericArray, FloatingArray], NumericArray]:
        """Return a compiled function for linear interpolation on the grid.

        Args:
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, `ValueError`
                is raised when out-of-bounds points are requested. Otherwise, the given
                value is returned.
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the ghost points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.
            cell_coords (bool):
                Flag indicating whether points are given in cell coordinates or actual
                point coordinates.

        Returns:
            callable: A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. The signature of this
            function is (data, point), where `data` is the numpy array containing the
            field data and position denotes the position in grid coordinates.
        """
        # deprecated on 2025-12-06
        from ..backends.numba.grids import make_single_interpolator

        warnings.warn(
            "`_make_interpolator_compiled` is deprecated. Use "
            "`pde.backends.numba_backend.make_single_interpolator` instead.",
            stacklevel=2,
        )
        return make_single_interpolator(
            grid=self,
            fill=fill,
            with_ghost_cells=with_ghost_cells,
            cell_coords=cell_coords,
        )

    def make_inserter_compiled(
        self, *, with_ghost_cells: bool = False
    ) -> Callable[[NumericArray, FloatingArray, NumberOrArray], None]:
        """Return a compiled function to insert values at interpolated positions.

        Args:
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the grid points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.

        Returns:
            callable: A function with signature (data, position, amount), where `data`
            is the numpy array containing the field data, position is denotes the
            position in grid coordinates, and `amount` is the  that is to be added to
            the field.
        """
        from ..backends.numba import numba_backend

        # deprecated since 2025-11-19
        warnings.warn(
            "`make_inserter_compiled` is deprecated. Use "
            "`pde.backends.numba_backend.make_inserter` instead",
            DeprecationWarning,
            stacklevel=2,
        )

        return numba_backend.make_inserter(grid=self, with_ghost_cells=with_ghost_cells)

    def make_integrator(
        self, backend: str = "numpy"
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that can be used to integrates discretized data over the
        grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            backend (str):
                The backend to use for making the operator

        Returns:
            callable: A function that takes a numpy array and returns the integral with
            the correct weights given by the cell volumes.
        """
        from ..backends import backends

        return backends[backend].make_integrator(self)


def registered_operators() -> dict[str, list[str]]:
    """Returns all operators that are currently defined.

    Returns:
        dict: a dictionary with the names of the operators defined for each grid class
    """
    return {
        name: sorted(cls.operators)
        for name, cls in GridBase._subclasses.items()
        if not (
            name.endswith("Base") or (hasattr(cls, "deprecated") and cls.deprecated)
        )
    }
