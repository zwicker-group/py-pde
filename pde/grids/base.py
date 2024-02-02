"""
Defines the base class for all grids

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import logging
import math
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, overload

import numba as nb
import numpy as np
from numba.extending import is_jitted
from numba.extending import overload as nb_overload
from numba.extending import register_jitable
from numpy.typing import ArrayLike

from ..tools.cache import cached_method, cached_property
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, hybridmethod
from ..tools.numba import jit
from ..tools.typing import (
    CellVolume,
    FloatNumerical,
    NumberOrArray,
    OperatorFactory,
    OperatorType,
)
from .coordinates import CoordinatesBase, DimensionError

if TYPE_CHECKING:
    from ._mesh import GridMesh
    from .boundaries.axes import Boundaries, BoundariesData


PI_4 = 4 * np.pi
PI_43 = 4 / 3 * np.pi
CoordsType = Literal["cartesian", "grid", "cell"]


class OperatorInfo(NamedTuple):
    """stores information about an operator"""

    factory: OperatorFactory
    rank_in: int
    rank_out: int
    name: str = ""  # attach a unique name to help caching


def _check_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    """checks the consistency of shape tuples"""
    if hasattr(shape, "__iter__"):
        shape_list: Sequence[int] = shape  # type: ignore
    else:
        shape_list = [shape]

    if len(shape_list) == 0:
        raise ValueError("Require at least one dimension")

    # convert the shape to a tuple of integers
    result = []
    for dim in shape_list:
        if dim == int(dim) and dim >= 1:
            result.append(int(dim))
        else:
            raise ValueError(f"{repr(dim)} is not a valid number of support points")
    return tuple(result)


def discretize_interval(
    x_min: float, x_max: float, num: int
) -> tuple[np.ndarray, float]:
    r"""construct a list of equidistantly placed intervals 

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
    """exception indicating that point lies outside domain"""


class PeriodicityError(RuntimeError):
    """exception indicating that the grid periodicity is inconsistent"""


class GridBase(metaclass=ABCMeta):
    """Base class for all grids defining common methods and interfaces"""

    # class properties
    _subclasses: dict[str, type[GridBase]] = {}  # all classes inheriting from this
    _operators: dict[str, OperatorInfo] = {}  # all operators defined for the grid

    # properties that are defined in subclasses
    c: CoordinatesBase
    """:class:`~pde.grids.coordinates.CoordinatesBase`: Coordinates of the grid"""
    axes: list[str]
    """list: Names of all axes that are described by the grid"""
    axes_symmetric: list[str] = []
    """list: The names of the additional axes that the fields do not depend on,
    e.g. along which they are constant. """

    boundary_names: dict[str, tuple[int, bool]] = {}
    """dict: Names of boundaries to select them conveniently"""
    cell_volume_data: Sequence[FloatNumerical] | None
    """list: Information about the size of discretization cells"""
    coordinate_constraints: list[int] = []
    """list: axes that not described explicitly"""
    num_axes: int
    """int: Number of axes that are *not* assumed symmetrically"""

    # mandatory, immutable, private attributes
    _axes_symmetric: tuple[int, ...] = ()
    _axes_described: tuple[int, ...]
    _axes_bounds: tuple[tuple[float, float], ...]
    _axes_coords: tuple[np.ndarray, ...]
    _discretization: np.ndarray
    _periodic: list[bool]
    _shape: tuple[int, ...]

    # to help sphinx, we here list docstrings for classproperties
    operators: set[str]
    """ set: names of all operators defined for this grid """

    def __init__(self) -> None:
        """initialize the grid"""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._mesh: GridMesh | None = None

        self._axes_described = tuple(
            i for i in range(self.dim) if i not in self._axes_symmetric
        )
        self.num_axes = len(self._axes_described)
        self.axes = [self.c.axes[i] for i in self._axes_described]
        self.axes_symmetric = [self.c.axes[i] for i in self.axes_symmetric]  # type: ignore

    def __init_subclass__(cls, **kwargs) -> None:  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        if cls is not GridBase:
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}")
            cls._subclasses[cls.__name__] = cls
        cls._operators = {}

    @classmethod
    def from_state(cls, state: str | dict[str, Any]) -> GridBase:
        """create a field from a stored `state`.

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
            raise RuntimeError(f"Cannot reconstruct abstract class `{class_name}`")
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
    def axes_coords(self) -> tuple[np.ndarray, ...]:
        """tuple: coordinates of the cells for each axis"""
        return self._axes_coords

    def get_axis_index(self, key: int | str, allow_symmetric: bool = True) -> int:
        """return the index belonging to an axis

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
            else:
                raise IndexError(f"`{key}` is not in the axes {axes}")
        elif isinstance(key, int):
            # assume that it is already an index
            return key
        raise IndexError("Index must be an integer or the name of an axes")

    def _get_boundary_index(self, index: str | tuple[int, bool]) -> tuple[int, bool]:
        """return the index of a boundary belonging to an axis

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
            axis, upper = self.boundary_names[index]
        else:
            axis, upper = index
        return axis, upper

    @property
    def discretization(self) -> np.ndarray:
        """:class:`numpy.array`: the linear size of a cell along each axis"""
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

    def _make_get_valid(self) -> Callable[[np.ndarray], np.ndarray]:
        """create a function to extract the valid part of a full data array

        Returns:
            callable: Mapping a numpy array containing the full data of the grid to a
                numpy array of only the valid data
        """
        num_axes = self.num_axes

        @jit
        def get_valid(data_full: np.ndarray) -> np.ndarray:
            """return valid part of the data (without ghost cells)

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The array with ghost cells from which the valid data is extracted
            """
            if num_axes == 1:
                return data_full[..., 1:-1]
            elif num_axes == 2:
                return data_full[..., 1:-1, 1:-1]
            elif num_axes == 3:
                return data_full[..., 1:-1, 1:-1, 1:-1]
            else:
                raise NotImplementedError

        return get_valid  # type: ignore

    @overload
    def _make_set_valid(self) -> Callable[[np.ndarray, np.ndarray], None]: ...

    @overload
    def _make_set_valid(
        self, bcs: Boundaries
    ) -> Callable[[np.ndarray, np.ndarray, dict], None]: ...

    def _make_set_valid(self, bcs: Boundaries | None = None) -> Callable:
        """create a function to set the valid part of a full data array

        Args:
            bcs (:class:`~pde.grids.boundaries.axes.Boundaries`, optional):
                If supplied, the returned function also enforces boundary conditions by
                setting the ghost cells to the correct values

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        num_axes = self.num_axes

        @jit
        def set_valid(data_full: np.ndarray, data_valid: np.ndarray) -> None:
            """set valid part of the data (without ghost cells)

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The full array with ghost cells that the data is written to
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to `data_full`
            """
            if num_axes == 1:
                data_full[..., 1:-1] = data_valid
            elif num_axes == 2:
                data_full[..., 1:-1, 1:-1] = data_valid
            elif num_axes == 3:
                data_full[..., 1:-1, 1:-1, 1:-1] = data_valid
            else:
                raise NotImplementedError

        if bcs is None:
            # just set the valid elements and leave ghost cells with arbitrary data_valids
            return set_valid  # type: ignore
        else:
            # set the valid elements and the ghost cells according to boundary condition
            set_bcs = bcs.make_ghost_cell_setter()

            @jit
            def set_valid_bcs(
                data_full: np.ndarray, data_valid: np.ndarray, args=None
            ) -> None:
                """set valid part of the data and the ghost cells using BCs

                Args:
                    data_full (:class:`~numpy.ndarray`):
                        The full array with ghost cells that the data is written to
                    data_valid (:class:`~numpy.ndarray`):
                        The valid data that is written to `data_full`
                    args (dict):
                        Extra arguments affecting the boundary conditions
                """
                set_valid(data_full, data_valid)
                set_bcs(data_full, args=args)

            return set_valid_bcs  # type: ignore

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
        """return a copy of the grid"""
        return self.__class__.from_state(self.state)

    __copy__ = copy

    def __deepcopy__(self, memo: dict[int, Any]) -> GridBase:
        """create a deep copy of the grid. This function is for instance called when
        a grid instance appears in another object that is copied using `copy.deepcopy`
        """
        # this implementation assumes that a simple call to copy is sufficient
        result = self.copy()
        memo[id(self)] = result
        return result

    def __repr__(self) -> str:
        """return instance as string"""
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
        """returns a value to determine when a cache needs to be updated"""
        return hash(
            (
                self.__class__.__name__,
                self.shape,
                self.axes_bounds,
                tuple(self.periodic),
            )
        )

    def compatible_with(self, other: GridBase) -> bool:
        """tests whether this grid is compatible with other grids.

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
        """checks whether `other` is compatible with the current grid

        Args:
            other (:class:`~pde.grids.base.GridBase`):
                The grid compared to this one

        Raises:
            ValueError: if grids are not compatible
        """
        if not self.compatible_with(other):
            raise ValueError(f"Grids {self} and {other} are incompatible")

    @property
    def numba_type(self) -> str:
        """str: represents type of the grid data in numba signatures"""
        return "f8[" + ", ".join([":"] * self.num_axes) + "]"

    @cached_property()
    def coordinate_arrays(self) -> tuple[np.ndarray, ...]:
        """tuple: for each axes: coordinate values for all cells"""
        return tuple(np.meshgrid(*self.axes_coords, indexing="ij"))

    @cached_property()
    def cell_coords(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: coordinate values for all axes of each cell"""
        return np.moveaxis(self.coordinate_arrays, 0, -1)

    @cached_property()
    def cell_volumes(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: volume of each cell"""
        if self.cell_volume_data is None:
            # use the self.c to calculate cell volumes
            d2 = self.discretization / 2
            x_low = self._coords_full(self.cell_coords - d2, value="min")
            x_high = self._coords_full(self.cell_coords + d2, value="max")
            return self.c.cell_volume(x_low, x_high)

        else:
            # use cell_volume_data
            vols = functools.reduce(np.outer, self.cell_volume_data)
            return np.broadcast_to(vols, self.shape)

    @cached_property()
    def uniform_cell_volumes(self) -> bool:
        """bool: returns True if all cell volumes are the same"""
        if self.cell_volume_data is None:
            return False
        else:
            return all(np.asarray(vols).ndim == 0 for vols in self.cell_volume_data)

    def _difference_vector(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        *,
        coords: CoordsType,
        periodic: Sequence[bool],
        axes_bounds: tuple[tuple[float, float], ...] | None,
    ) -> np.ndarray:
        """return Cartesian vector(s) pointing from p1 to p2

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
        return diff  # type: ignore

    def difference_vector(
        self, p1: np.ndarray, p2: np.ndarray, *, coords: CoordsType = "grid"
    ) -> np.ndarray:
        """return Cartesian vector(s) pointing from p1 to p2

        In case of periodic boundary conditions, the shortest vector is returned.

        Args:
            p1 (:class:`~numpy.ndarray`):
                First point(s)
            p2 (:class:`~numpy.ndarray`):
                Second point(s)
            coords (str):
                The coordinate system in which the points are specified. Valid values are
                `cartesian`, `cell`, and `grid`; see :meth:`~pde.grids.base.GridBase.transform`.

        Returns:
            :class:`~numpy.ndarray`: The difference vectors between the points with
            periodic boundary conditions applied.
        """
        return self._difference_vector(
            p1, p2, coords=coords, periodic=[False] * self.dim, axes_bounds=None
        )

    def difference_vector_real(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # deprecated on 2024-01-09
        warnings.warn(
            "`difference_vector_real` has been renamed to `difference_vector`",
            DeprecationWarning,
        )
        return self.difference_vector(p1, p2)

    def distance(
        self, p1: np.ndarray, p2: np.ndarray, *, coords: CoordsType = "grid"
    ) -> float:
        """Calculate the distance between two points given in real coordinates

        This takes periodic boundary conditions into account if necessary.

        Args:
            p1 (:class:`~numpy.ndarray`):
                First position
            p2 (:class:`~numpy.ndarray`):
                Second position
            coords (str):
                The coordinate system in which the points are specified. Valid values are
                `cartesian`, `cell`, and `grid`; see :meth:`~pde.grids.base.GridBase.transform`.

        Returns:
            float: Distance between the two positions
        """
        diff = self.difference_vector(p1, p2, coords=coords)
        return np.linalg.norm(diff, axis=-1)  # type: ignore

    def distance_real(self, p1: np.ndarray, p2: np.ndarray) -> float:
        # deprecated on 2024-01-09
        warnings.warn(
            "`distance_real` has been renamed to `distance`",
            DeprecationWarning,
        )
        return self.distance(p1, p2)

    def _iter_boundaries(self) -> Iterator[tuple[int, bool]]:
        """iterate over all boundaries of the grid

        Yields:
            tuple: for each boundary, the generator returns a tuple indicating
            the axis of the boundary together with a boolean value indicating
            whether the boundary lies on the upper side of the axis.
        """
        return itertools.product(range(self.num_axes), [True, False])

    def _boundary_coordinates(
        self, axis: int, upper: bool, *, offset: float = 0
    ) -> np.ndarray:
        """get coordinates of points on the boundary

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
        shape = shape_bndry + (self.num_axes,)
        return np.stack(points, -1).reshape(shape)  # type: ignore

    @property
    def volume(self) -> float:
        """float: total volume of the grid"""
        # this property should be overwritten when the volume can be calculated directly
        return self.cell_volumes.sum()  # type: ignore

    def point_to_cartesian(
        self, points: np.ndarray, *, full: bool = False
    ) -> np.ndarray:
        """convert coordinates of a point in grid coordinates to Cartesian coordinates

        Args:
            points (:class:`~numpy.ndarray`):
                The grid coordinates of the points
            full (bool):
                Indicates whether coordinates along symmetric axes are specified

        Returns:
            :class:`~numpy.ndarray`: The Cartesian coordinates of the point
        """
        if full:
            # Deprecated on 2024-01-31
            warnings.warn(
                "`full=True` is deprecated. Use `grid.c.pos_to_cart` instead",
                DeprecationWarning,
            )
        else:
            points = self._coords_full(points)
        return self.c.pos_to_cart(points)

    def point_from_cartesian(
        self, points: np.ndarray, *, full: bool = False
    ) -> np.ndarray:
        """convert points given in Cartesian coordinates to grid coordinates

        Args:
            points (:class:`~numpy.ndarray`):
                Points given in Cartesian coordinates.
            full (bool):
                Indicates whether coordinates along symmetric axes are specified

        Returns:
            :class:`~numpy.ndarray`: Points given in the coordinates of the grid
        """
        points_sph = self.c.pos_from_cart(points)
        if full:
            # Deprecated since 2024-01-31
            warnings.warn(
                "`full=True` is deprecated. Use `grid.c.pos_from_cart` instead",
                DeprecationWarning,
            )
            return points_sph
        else:
            return self._coords_symmetric(points_sph)

    def _vector_to_cartesian(
        self, points: ArrayLike, components: ArrayLike
    ) -> np.ndarray:
        """convert the vectors at given points into a Cartesian basis

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
            raise DimensionError(f"`points` must have {self.dim} coordinates")
        shape = points.shape[:-1]  # shape of array describing the different points
        vec_shape = (self.dim,) + shape
        if components.shape != vec_shape:
            raise DimensionError(f"`components` must have shape {vec_shape}")

        # convert the basis of the vectors to Cartesian
        rot_mat = self.c.basis_rotation(points)
        assert rot_mat.shape == (self.dim, self.dim) + shape
        return np.einsum("j...,ji...->i...", components, rot_mat)  # type: ignore

    def normalize_point(
        self, point: np.ndarray, *, reflect: bool = False
    ) -> np.ndarray:
        """normalize grid coordinates by applying periodic boundary conditions

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
                raise DimensionError(
                    f"Point {point} is not of dimension {self.num_axes}"
                )
        elif point.shape[-1] != self.num_axes:
            raise DimensionError(
                f"Array of shape {point.shape} does not describe points of dimension "
                f"{self.num_axes}"
            )

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

    def _coords_symmetric(self, points: np.ndarray) -> np.ndarray:
        """return only non-symmetric point coordinates

        Args:
            points (:class:`~numpy.ndarray`):
                The points specified with `dim` coordinates

        Returns:
            :class:`~numpy.ndarray`: The points with only `num_axes` coordinates, which
            are not along symmetry axes of the grid.
        """
        if points.shape[-1] != self.dim:
            raise DimensionError(f"Points need to be specified as {self.c.axes}")
        return points[..., self._axes_described]

    def _coords_full(
        self, points: np.ndarray, *, value: Literal["min", "max"] | float = 0.0
    ) -> np.ndarray:
        """specify point coordinates along symmetric axes on grids

        Args:
            points (:class:`~numpy.ndarray`):
                The points specified with `num_axes` coordinates, not specifying
                cooridnates along symmetry axes of the grid.
            value (str or float):
                Value of the points along symmetry axes. The special values `min` and
                `max` denote the minimal and maximal values along the respective
                coordinates.

        Returns:
            :class:`~numpy.ndarray`: The points with all `dim` coordinates

        """
        if self.num_axes == self.dim:
            return points
        else:
            if points.shape[-1] != self.num_axes:
                raise DimensionError(f"Points need to be specified as {self.axes}")
            res = np.empty(points.shape[:-1] + (self.dim,), dtype=points.dtype)
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
        self,
        coordinates: np.ndarray,
        source: CoordsType,
        target: CoordsType,
        *,
        full: bool = False,
    ) -> np.ndarray:
        """converts coordinates from one coordinate system to another

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
            full (bool):
                Indicates whether coordinates along symmetric axes are specified

        Returns:
            :class:`~numpy.ndarray`: The transformed coordinates
        """
        if full:
            # Deprecated since 2024-01-31
            warnings.warn(
                "`full=True` is deprecated. Use `grid.c` methods instead",
                DeprecationWarning,
            )

        if source == "cartesian":
            # Cartesian coordinates given
            cartesian = np.atleast_1d(coordinates)
            if cartesian.shape[-1] != self.dim:
                raise DimensionError(f"Require {self.dim} cartesian coordinates")

            if target == "cartesian":
                return coordinates

            # convert Cartesian coordinates to grid coordinates
            grid_coords = self.point_from_cartesian(cartesian, full=full)

            if target == "grid":
                return grid_coords
            if target == "cell":
                c_min = np.array(self.axes_bounds)[:, 0]
                if full:
                    # remove the coordinates that are symmetric
                    grid_coords = self._coords_symmetric(grid_coords)
                return (grid_coords - c_min) / self.discretization  # type: ignore

        elif source == "cell":
            # Cell coordinates given
            if full:
                raise ValueError("Cell coordinates cannot be given with `full=True`")

            cells = np.atleast_1d(coordinates)
            if cells.shape[-1] != self.num_axes:
                raise DimensionError(f"Require {self.num_axes} cell coordinates")

            if target == "cell":
                return coordinates

            # convert cell coordinates to grid coordinates
            c_min = np.array(self.axes_bounds)[:, 0]
            grid_coords = c_min + cells * self.discretization

            if target == "grid":
                return grid_coords
            elif target == "cartesian":
                return self.point_to_cartesian(grid_coords, full=False)

        elif source == "grid":
            # Grid coordinates given
            grid_coords = np.atleast_1d(coordinates)
            if full and grid_coords.shape[-1] != self.dim:
                raise DimensionError(f"Require {self.dim} grid coordinates")
            if not full and grid_coords.shape[-1] != self.num_axes:
                raise DimensionError(f"Require {self.num_axes} grid coordinates")

            if target == "cartesian":
                return self.point_to_cartesian(grid_coords, full=full)
            elif target == "cell":
                c_min = np.array(self.axes_bounds)[:, 0]
                if full:
                    # remove the coordinates that are symmetric
                    grid_coords = self._coords_symmetric(grid_coords)
                return (grid_coords - c_min) / self.discretization  # type: ignore
            elif target == "grid":
                return grid_coords

        else:
            raise ValueError(f"Unknown source coordinates `{source}`")
        raise ValueError(f"Unknown target coordinates `{target}`")

    def contains_point(
        self,
        points: np.ndarray,
        *,
        coords: Literal["cartesian", "cell", "grid"] = "cartesian",
        full: bool = False,
    ) -> np.ndarray:
        """check whether the point is contained in the grid

        Args:
            point (:class:`~numpy.ndarray`):
                Coordinates of the point
            coords (str):
                The coordinate system in which the points are given
            full (bool):
                Indicates whether coordinates along symmetric axes are specified

        Returns:
            :class:`~numpy.ndarray`: A boolean array indicating which points lie within
            the grid
        """
        cell_coords = self.transform(points, source=coords, target="cell", full=full)
        return np.all((0 <= cell_coords) & (cell_coords <= self.shape), axis=-1)  # type: ignore

    def iter_mirror_points(
        self, point: np.ndarray, with_self: bool = False, only_periodic: bool = True
    ) -> Generator:
        """generates all mirror points corresponding to `point`

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
    ) -> Boundaries:
        """constructs boundary conditions from a flexible data format

        Args:
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            rank (int):
                The tensorial rank of the value associated with the boundary conditions.

        Returns:
            :class:`~pde.grids.boundaries.axes.Boundaries`: The boundary conditions for
            all axes.

        Raises:
            ValueError:
                If the data given in `bc` cannot be read
            PeriodicityError:
                If the boundaries are not compatible with the periodic axes of the grid.
        """
        from .boundaries import Boundaries  # @Reimport

        if self._mesh is None:
            # get boundary conditions for a simple grid that is not part of a mesh
            bcs = Boundaries.from_data(self, bc, rank=rank)

        else:
            # this grid is part of a mesh and we thus need to set special conditions to
            # support parallelism via MPI. We here assume that bc is given for the full
            # system and not
            bcs_base = Boundaries.from_data(self._mesh.basegrid, bc, rank=rank)
            bcs = self._mesh.extract_boundary_conditions(bcs_base)

        return bcs

    def get_line_data(self, data: np.ndarray, extract: str = "auto") -> dict[str, Any]:
        """return a line cut through the grid

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

    def get_image_data(self, data: np.ndarray) -> dict[str, Any]:
        """return a 2d-image of the data

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points

        Returns:
            dict: A dictionary with information about the data convenient for plotting.
        """
        raise NotImplementedError

    def get_vector_data(self, data: np.ndarray, **kwargs) -> dict[str, Any]:
        r"""return data to visualize vector field

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
            raise DimensionError("Can only plot generic vector fields for dim=2")
        if data.shape != (self.dim,) + self.shape:
            raise ValueError(
                f"Shape {data.shape} of the data array is not compatible with grid "
                f"shape {self.shape}"
            )

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
    ) -> np.ndarray:
        """return a random point within the grid

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

    @classmethod
    def register_operator(
        cls,
        name: str,
        factory_func: OperatorFactory | None = None,
        rank_in: int = 0,
        rank_out: int = 0,
    ):
        """register an operator for this grid

        Example:
            The method can either be used directly:

            .. code-block:: python

                GridClass.register_operator("operator", make_operator)

            or as a decorator for the factory function:

            .. code-block:: python

                @GridClass.register_operator("operator")
                def make_operator(grid: GridBase):
                    ...

        Args:
            name (str):
                The name of the operator to register
            factory_func (callable):
                A function with signature ``(grid: GridBase, **kwargs)``, which takes
                a grid object and optional keyword arguments and returns an
                implementation of the given operator. This implementation is a function
                that takes a :class:`~numpy.ndarray` of discretized values as arguments
                and returns the resulting discretized data in a :class:`~numpy.ndarray`
                after applying the operator.
            rank_in (int):
                The rank of the input field for the operator
            rank_out (int):
                The rank of the field that is returned by the operator
        """

        def register_operator(factor_func_arg: OperatorFactory):
            """helper function to register the operator"""
            cls._operators[name] = OperatorInfo(
                factory=factor_func_arg, rank_in=rank_in, rank_out=rank_out, name=name
            )
            return factor_func_arg

        if factory_func is None:
            # method is used as a decorator, so return the helper function
            return register_operator
        else:
            # method is used directly
            register_operator(factory_func)

    @hybridmethod  # type: ignore
    @property
    def operators(cls) -> set[str]:  # @NoSelf
        """set: all operators defined for this class"""
        result = set()
        # add all customly defined operators
        classes = inspect.getmro(cls)[:-1]  # type: ignore
        for anycls in classes:
            result |= set(anycls._operators.keys())  # type: ignore
        if hasattr(cls, "axes"):
            for ax in cls.axes:
                result |= {
                    f"d_d{ax}",
                    f"d_d{ax}_forward",
                    f"d_d{ax}_backward",
                    f"d2_d{ax}2",
                }
        return result

    @operators.instancemethod
    @property
    def operators(self) -> set[str]:
        """set: all operators defined for this instance"""
        # get all operators registered on the class
        result = self.__class__.operators
        if not hasattr(self.__class__, "axes"):
            # add operators calculating derivate along a coordinate for the case where
            # the axes argument is only defined on instances
            for ax in self.axes:
                result |= {f"d_d{ax}", f"d2_d{ax}2"}
        return result

    def _get_operator_info(self, operator: str | OperatorInfo) -> OperatorInfo:
        """return the operator defined on this grid

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.

        Returns:
            :class:`~pde.grids.base.OperatorInfo`: information for the operator
        """
        if isinstance(operator, OperatorInfo):
            return operator
        assert isinstance(operator, str)

        # look for defined operators on all parent classes (except `object`)
        classes = inspect.getmro(self.__class__)[:-1]
        for cls in classes:
            if operator in cls._operators:  # type: ignore
                return cls._operators[operator]  # type: ignore

        # deal with some special patterns that are often used
        if operator.startswith("d_d"):
            # create a special operator that takes a first derivative along one axis
            from .operators.common import make_derivative

            # determine axis to which operator is applied (and the method to use)
            axis_name = operator[len("d_d") :]
            for direction in ["central", "forward", "backward"]:
                if axis_name.endswith("_" + direction):
                    method = direction
                    axis_name = axis_name[: -len("_" + direction)]
                    break
            else:
                method = "central"

            axis_id = self.axes.index(axis_name)
            factory = functools.partial(make_derivative, axis=axis_id, method=method)
            return OperatorInfo(factory, rank_in=0, rank_out=0, name=operator)

        elif operator.startswith("d2_d") and operator.endswith("2"):
            # create a special operator that takes a second derivative along one axis
            from .operators.common import make_derivative2

            axis_id = self.axes.index(operator[len("d2_d") : -1])
            factory = functools.partial(make_derivative2, axis=axis_id)
            return OperatorInfo(factory, rank_in=0, rank_out=0, name=operator)

        # throw an informative error since operator was not found
        op_list = ", ".join(sorted(self.operators))
        raise ValueError(
            f"'{operator}' is not one of the defined operators ({op_list}). Custom "
            "operators can be added using the `register_operator` method."
        )

    @cached_method()
    def make_operator_no_bc(
        self,
        operator: str | OperatorInfo,
        **kwargs,
    ) -> OperatorType:
        """return a compiled function applying an operator without boundary conditions

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
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: np.ndarray, out: np.ndarray), so they `out` array need to be
            supplied explicitly.
        """
        return self._get_operator_info(operator).factory(self, **kwargs)

    @cached_method()
    @fill_in_docstring
    def make_operator(
        self, operator: str | OperatorInfo, bc: BoundariesData, **kwargs
    ) -> Callable[..., np.ndarray]:
        """return a compiled function applying an operator with boundary conditions

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array. The function also accepts an
        optional parameter `args`, which is forwarded to `set_ghost_cells`.

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: np.ndarray, out: np.ndarray = None, args=None).
        """
        backend = kwargs.get("backend", "numba")  # numba is the default backend

        # instantiate the operator
        operator = self._get_operator_info(operator)
        operator_raw = operator.factory(self, **kwargs)

        # set the boundary conditions before applying this operator
        bcs = self.get_boundary_conditions(bc, rank=operator.rank_in)

        # calculate shapes of the full data
        shape_in_valid = (self.dim,) * operator.rank_in + self.shape
        shape_in_full = (self.dim,) * operator.rank_in + self._shape_full
        shape_out = (self.dim,) * operator.rank_out + self.shape

        # define numpy version of the operator
        def apply_op(
            arr: np.ndarray, out: np.ndarray | None = None, args=None
        ) -> np.ndarray:
            """set boundary conditions and apply operator"""
            assert arr.shape == shape_in_valid
            # ensure `out` array is allocated
            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            else:
                assert out.shape == shape_out

            # prepare input with boundary conditions
            arr_full = np.empty(shape_in_full, dtype=arr.dtype)
            arr_full[(...,) + self._idx_valid] = arr
            bcs.set_ghost_cells(arr_full, args=args)

            # apply operator
            operator_raw(arr_full, out)

            # return valid part of the output
            return out

        if backend in {"numpy", "scipy"}:
            # return the bare operator without the numba-overloaded version
            return apply_op

        elif backend.startswith("numba"):
            # overload `apply_op` with numba-compiled version
            # set_ghost_cells = bcs.make_ghost_cell_setter()
            set_valid_w_bc = self._make_set_valid(bcs=bcs)

            if not is_jitted(operator_raw):
                operator_raw = jit(operator_raw)

            @nb_overload(apply_op, inline="always")
            def apply_op_ol(
                arr: np.ndarray, out: np.ndarray | None = None, args=None
            ) -> np.ndarray:
                """make numba implementation of the operator"""
                if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                    # need to allocate memory for `out`

                    def apply_op_impl(
                        arr: np.ndarray, out: np.ndarray | None = None, args=None
                    ) -> np.ndarray:
                        """allocates `out` and applies operator to the data"""
                        assert arr.shape == shape_in_valid

                        out = np.empty(shape_out, dtype=arr.dtype)
                        # prepare input with boundary conditions
                        arr_full = np.empty(shape_in_full, dtype=arr.dtype)
                        set_valid_w_bc(arr_full, arr, args=args)  # type: ignore

                        # apply operator
                        operator_raw(arr_full, out)

                        # return valid part of the output
                        return out

                else:
                    # reuse provided `out` array

                    def apply_op_impl(
                        arr: np.ndarray, out: np.ndarray | None = None, args=None
                    ) -> np.ndarray:
                        """applies operator to the data wihtout allocating out"""
                        assert arr.shape == shape_in_valid
                        assert out.shape == shape_out  # type: ignore

                        # prepare input with boundary conditions
                        arr_full = np.empty(shape_in_full, dtype=arr.dtype)
                        set_valid_w_bc(arr_full, arr, args=args)  # type: ignore

                        # apply operator
                        operator_raw(arr_full, out)  # type: ignore

                        # return valid part of the output
                        return out  # type: ignore

                return apply_op_impl  # type: ignore

            @jit
            def apply_op_compiled(
                arr: np.ndarray, out: np.ndarray | None = None, args=None
            ) -> np.ndarray:
                """set boundary conditions and apply operator"""
                return apply_op(arr, out, args)

            # return the compiled versions of the operator
            return apply_op_compiled  # type: ignore

        else:
            # simply return the operator if the backend was `numba` or `scipy`
            raise NotImplementedError(f"Undefined backend '{backend}'")

    def slice(self, indices: Sequence[int]) -> GridBase:
        """return a subgrid of only the specified axes

        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid

        Returns:
            :class:`GridBase`: The subgrid
        """
        raise NotImplementedError(
            f"Slicing is not implemented for class {self.__class__.__name__}"
        )

    def plot(self) -> None:
        """visualize the grid"""
        raise NotImplementedError(
            f"Plotting is not implemented for class {self.__class__.__name__}"
        )

    @property
    def typical_discretization(self) -> float:
        """float: the average side length of the cells"""
        return np.mean(self.discretization)  # type: ignore

    def integrate(
        self, data: NumberOrArray, axes: int | Sequence[int] | None = None
    ) -> NumberOrArray:
        """Integrates the discretized data over the grid

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

        else:
            # we are in a parallel run, so we need to gather the sub-integrals from all
            from mpi4py.MPI import COMM_WORLD  # @UnresolvedImport

            integral_full = np.empty_like(integral)
            COMM_WORLD.Allreduce(integral, integral_full)
            return integral_full  # type: ignore

    @cached_method()
    def make_normalize_point_compiled(
        self, reflect: bool = True
    ) -> Callable[[np.ndarray], None]:
        """return a compiled function that normalizes a point

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
        num_axes = self.num_axes
        periodic = np.array(self.periodic)  # using a tuple instead led to a numba error
        bounds = np.array(self.axes_bounds)
        xmin = bounds[:, 0]
        xmax = bounds[:, 1]
        size = bounds[:, 1] - bounds[:, 0]

        @jit
        def normalize_point(point: np.ndarray) -> None:
            """helper function normalizing a single point"""
            assert point.ndim == 1  # only support single points
            for i in range(num_axes):
                if periodic[i]:
                    point[i] = (point[i] - xmin[i]) % size[i] + xmin[i]
                elif reflect:
                    arg = (point[i] - xmax[i]) % (2 * size[i]) - size[i]
                    point[i] = xmin[i] + abs(arg)
                # else: do nothing

        return normalize_point  # type: ignore

    @cached_method()
    def make_cell_volume_compiled(self, flat_index: bool = False) -> CellVolume:
        """return a compiled function returning the volume of a grid cell

        Args:
            flat_index (bool):
                When True, cell_volumes are indexed by a single integer into the
                flattened array.

        Returns:
            function: returning the volume of the chosen cell
        """
        if self.cell_volume_data is not None and all(
            np.isscalar(d) for d in self.cell_volume_data
        ):
            # all cells have the same volume
            cell_volume = np.prod(self.cell_volume_data)  # type: ignore

            @jit
            def get_cell_volume(*args) -> float:
                return cell_volume  # type: ignore

        else:
            # some cells have a different volume
            cell_volumes = self.cell_volumes

            if flat_index:

                @jit
                def get_cell_volume(idx: int) -> float:
                    return cell_volumes.flat[idx]  # type: ignore

            else:

                @jit
                def get_cell_volume(*args) -> float:
                    return cell_volumes[args]  # type: ignore

        return get_cell_volume  # type: ignore

    def _make_interpolation_axis_data(
        self,
        axis: int,
        *,
        with_ghost_cells: bool = False,
        cell_coords: bool = False,
    ) -> Callable[[float], tuple[int, int, float, float]]:
        """factory for obtaining interpolation information

        Args:
            axis (int):
                The axis along which interpolation is performed
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the ghost points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.
            cell_coords (bool):
                Flag indicating whether points are given in cell coordinates or actual
                point coordinates.

        Returns:
            callable: A function that is called with a coordinate value for the axis.
            The function returns the indices of the neighboring support points as well
            as the associated weights.
        """
        # obtain information on how this axis is discretized
        size = self.shape[axis]
        periodic = self.periodic[axis]
        lo = self.axes_bounds[axis][0]
        dx = self.discretization[axis]

        @register_jitable
        def get_axis_data(coord: float) -> tuple[int, int, float, float]:
            """determines data for interpolating along one axis"""
            # determine the index of the left cell and the fraction toward the right
            if cell_coords:
                c_l, d_l = divmod(coord, 1.0)
            else:
                c_l, d_l = divmod((coord - lo) / dx - 0.5, 1.0)

            # determine the indices of the two cells whose value affect interpolation
            if periodic:
                # deal with periodic domains, which is easy
                c_li = int(c_l) % size  # left support point
                c_hi = (c_li + 1) % size  # right support point

            elif with_ghost_cells:
                # deal with edge cases using the values of ghost cells
                if -0.5 <= c_l + d_l <= size - 0.5:  # in bulk part of domain
                    c_li = int(c_l)  # left support point
                    c_hi = c_li + 1  # right support point
                else:
                    return -42, -42, 0.0, 0.0  # indicates out of bounds

            else:
                # deal with edge cases using nearest-neighbor interpolation at boundary
                if 0 <= c_l + d_l < size - 1:  # in bulk part of domain
                    c_li = int(c_l)  # left support point
                    c_hi = c_li + 1  # right support point
                elif size - 1 <= c_l + d_l <= size - 0.5:  # close to upper boundary
                    c_li = c_hi = int(c_l)  # both support points close to boundary
                    # This branch also covers the special case, where size == 1 and data
                    # is evaluated at the only support point (c_l == d_l == 0.)
                elif -0.5 <= c_l + d_l <= 0:  # close to lower boundary
                    c_li = c_hi = int(c_l) + 1  # both support points close to boundary
                else:
                    return -42, -42, 0.0, 0.0  # indicates out of bounds

            # determine the weights of the two cells
            w_l, w_h = 1 - d_l, d_l
            # set small weights to zero. If this is not done, invalid data at the corner
            # of the grid (where two rows of ghost cells intersect) could be accessed.
            # If this random data is very large, e.g., 1e100, it contributes
            # significantly, even if the weight is low, e.g., 1e-16.
            if w_l < 1e-15:
                w_l = 0
            if w_h < 1e-15:
                w_h = 0

            # shift points to allow accessing data with ghost points
            if with_ghost_cells:
                c_li += 1
                c_hi += 1

            return c_li, c_hi, w_l, w_h

        return get_axis_data  # type: ignore

    @cached_method()
    def _make_interpolator_compiled(
        self,
        *,
        fill: Number | None = None,
        with_ghost_cells: bool = False,
        cell_coords: bool = False,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """return a compiled function for linear interpolation on the grid

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
        args = {"with_ghost_cells": with_ghost_cells, "cell_coords": cell_coords}

        if self.num_axes == 1:
            # specialize for 1-dimensional interpolation
            data_x = self._make_interpolation_axis_data(0, **args)

            @jit
            def interpolate_single(
                data: np.ndarray, point: np.ndarray
            ) -> NumberOrArray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`~numpy.ndarray`):
                        A 1d array of valid values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system

                Returns:
                    :class:`~numpy.ndarray`: The interpolated value at the point
                """
                c_li, c_hi, w_l, w_h = data_x(point[0])

                if c_li == -42:  # out of bounds
                    if fill is None:  # outside the domain
                        print("POINT", point)
                        raise DomainError("Point lies outside the grid domain")
                    else:
                        return fill

                # do the linear interpolation
                return w_l * data[..., c_li] + w_h * data[..., c_hi]

        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            data_x = self._make_interpolation_axis_data(0, **args)
            data_y = self._make_interpolation_axis_data(1, **args)

            @jit
            def interpolate_single(
                data: np.ndarray, point: np.ndarray
            ) -> NumberOrArray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`~numpy.ndarray`):
                        A 2d array of valid values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system

                Returns:
                    :class:`~numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(point[0])
                c_yli, c_yhi, w_yl, w_yh = data_y(point[1])

                if c_xli == -42 or c_yli == -42:  # out of bounds
                    if fill is None:  # outside the domain
                        print("POINT", point)
                        raise DomainError("Point lies outside the grid domain")
                    else:
                        return fill

                # do the linear interpolation
                return (  # type: ignore
                    w_xl * w_yl * data[..., c_xli, c_yli]
                    + w_xl * w_yh * data[..., c_xli, c_yhi]
                    + w_xh * w_yl * data[..., c_xhi, c_yli]
                    + w_xh * w_yh * data[..., c_xhi, c_yhi]
                )

        elif self.num_axes == 3:
            # specialize for 3-dimensional interpolation
            data_x = self._make_interpolation_axis_data(0, **args)
            data_y = self._make_interpolation_axis_data(1, **args)
            data_z = self._make_interpolation_axis_data(2, **args)

            @jit
            def interpolate_single(
                data: np.ndarray, point: np.ndarray
            ) -> NumberOrArray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`~numpy.ndarray`):
                        A 2d array of valid values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system

                Returns:
                    :class:`~numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(point[0])
                c_yli, c_yhi, w_yl, w_yh = data_y(point[1])
                c_zli, c_zhi, w_zl, w_zh = data_z(point[2])

                if c_xli == -42 or c_yli == -42 or c_zli == -42:  # out of bounds
                    if fill is None:  # outside the domain
                        print("POINT", point)
                        raise DomainError("Point lies outside the grid domain")
                    else:
                        return fill

                # do the linear interpolation
                return (  # type: ignore
                    w_xl * w_yl * w_zl * data[..., c_xli, c_yli, c_zli]
                    + w_xl * w_yl * w_zh * data[..., c_xli, c_yli, c_zhi]
                    + w_xl * w_yh * w_zl * data[..., c_xli, c_yhi, c_zli]
                    + w_xl * w_yh * w_zh * data[..., c_xli, c_yhi, c_zhi]
                    + w_xh * w_yl * w_zl * data[..., c_xhi, c_yli, c_zli]
                    + w_xh * w_yl * w_zh * data[..., c_xhi, c_yli, c_zhi]
                    + w_xh * w_yh * w_zl * data[..., c_xhi, c_yhi, c_zli]
                    + w_xh * w_yh * w_zh * data[..., c_xhi, c_yhi, c_zhi]
                )

        else:
            raise NotImplementedError(
                f"Compiled interpolation not implemented for dimension {self.num_axes}"
            )

        return interpolate_single  # type: ignore

    def make_inserter_compiled(
        self, *, with_ghost_cells: bool = False
    ) -> Callable[[np.ndarray, np.ndarray, NumberOrArray], None]:
        """return a compiled function to insert values at interpolated positions

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
        cell_volume = self.make_cell_volume_compiled()

        if self.num_axes == 1:
            # specialize for 1-dimensional interpolation
            data_x = self._make_interpolation_axis_data(
                0, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: np.ndarray, point: np.ndarray, amount: NumberOrArray
            ) -> None:
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                c_li, c_hi, w_l, w_h = data_x(point[0])

                if c_li == -42:  # out of bounds
                    raise DomainError("Point lies outside the grid domain")

                data[..., c_li] += w_l * amount / cell_volume(c_li)
                data[..., c_hi] += w_h * amount / cell_volume(c_hi)

        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            data_x = self._make_interpolation_axis_data(
                0, with_ghost_cells=with_ghost_cells
            )
            data_y = self._make_interpolation_axis_data(
                1, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: np.ndarray, point: np.ndarray, amount: NumberOrArray
            ) -> None:
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(point[0])
                c_yli, c_yhi, w_yl, w_yh = data_y(point[1])

                if c_xli == -42 or c_yli == -42:  # out of bounds
                    raise DomainError("Point lies outside the grid domain")

                cell_vol = cell_volume(c_xli, c_yli)
                data[..., c_xli, c_yli] += w_xl * w_yl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yhi)
                data[..., c_xli, c_yhi] += w_xl * w_yh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yli)
                data[..., c_xhi, c_yli] += w_xh * w_yl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yhi)
                data[..., c_xhi, c_yhi] += w_xh * w_yh * amount / cell_vol

        elif self.num_axes == 3:
            # specialize for 3-dimensional interpolation
            data_x = self._make_interpolation_axis_data(
                0, with_ghost_cells=with_ghost_cells
            )
            data_y = self._make_interpolation_axis_data(
                1, with_ghost_cells=with_ghost_cells
            )
            data_z = self._make_interpolation_axis_data(
                2, with_ghost_cells=with_ghost_cells
            )

            @jit
            def insert(
                data: np.ndarray, point: np.ndarray, amount: NumberOrArray
            ) -> None:
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`~numpy.ndarray`):
                        The values at the grid points
                    point (:class:`~numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate system
                    amount (Number or :class:`~numpy.ndarray`):
                        The amount that will be added to the data. This value describes
                        an integrated quantity (given by the field value times the
                        discretization volume). This is important for consistency with
                        different discretizations and in particular grids with
                        non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_xli, c_xhi, w_xl, w_xh = data_x(point[0])
                c_yli, c_yhi, w_yl, w_yh = data_y(point[1])
                c_zli, c_zhi, w_zl, w_zh = data_z(point[2])

                if c_xli == -42 or c_yli == -42 or c_zli == -42:  # out of bounds
                    raise DomainError("Point lies outside the grid domain")

                cell_vol = cell_volume(c_xli, c_yli, c_zli)
                data[..., c_xli, c_yli, c_zli] += w_xl * w_yl * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yli, c_zhi)
                data[..., c_xli, c_yli, c_zhi] += w_xl * w_yl * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xli, c_yhi, c_zli)
                data[..., c_xli, c_yhi, c_zli] += w_xl * w_yh * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xli, c_yhi, c_zhi)
                data[..., c_xli, c_yhi, c_zhi] += w_xl * w_yh * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yli, c_zli)
                data[..., c_xhi, c_yli, c_zli] += w_xh * w_yl * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yli, c_zhi)
                data[..., c_xhi, c_yli, c_zhi] += w_xh * w_yl * w_zh * amount / cell_vol

                cell_vol = cell_volume(c_xhi, c_yhi, c_zli)
                data[..., c_xhi, c_yhi, c_zli] += w_xh * w_yh * w_zl * amount / cell_vol
                cell_vol = cell_volume(c_xhi, c_yhi, c_zhi)
                data[..., c_xhi, c_yhi, c_zhi] += w_xh * w_yh * w_zh * amount / cell_vol

        else:
            raise NotImplementedError(
                f"Compiled interpolation not implemented for dimension {self.num_axes}"
            )

        return insert  # type: ignore

    def make_integrator(self) -> Callable[[np.ndarray], NumberOrArray]:
        """return function that can be used to integrates discretized data over the grid

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Returns:
            callable: A function that takes a numpy array and returns the integral with
            the correct weights given by the cell volumes.
        """
        num_axes = self.num_axes
        # cell volume varies with position
        get_cell_volume = self.make_cell_volume_compiled(flat_index=True)

        def integrate_local(arr: np.ndarray) -> NumberOrArray:
            """integrates data over a grid using numpy"""
            amounts = arr * self.cell_volumes
            return amounts.sum(axis=tuple(range(-num_axes, 0, 1)))  # type: ignore

        @nb_overload(integrate_local)
        def ol_integrate_local(
            arr: np.ndarray,
        ) -> Callable[[np.ndarray], NumberOrArray]:
            """integrates data over a grid using numba"""
            if arr.ndim == num_axes:
                # `arr` is a scalar field
                grid_shape = self.shape

                def impl(arr: np.ndarray) -> Number:
                    """integrate a scalar field"""
                    assert arr.shape == grid_shape
                    total = 0
                    for i in range(arr.size):
                        total += get_cell_volume(i) * arr.flat[i]
                    return total

            else:
                # `arr` is a tensorial field with rank >= 1
                tensor_shape = (self.dim,) * (arr.ndim - num_axes)
                data_shape = tensor_shape + self.shape

                def impl(arr: np.ndarray) -> np.ndarray:  # type: ignore
                    """integrate a tensorial field"""
                    assert arr.shape == data_shape
                    total = np.zeros(tensor_shape)
                    for idx in np.ndindex(*tensor_shape):
                        arr_comp = arr[idx]
                        for i in range(arr_comp.size):
                            total[idx] += get_cell_volume(i) * arr_comp.flat[i]
                    return total

            return impl

        # deal with MPI multiprocessing
        if self._mesh is None or len(self._mesh) == 1:
            # standard case of a single integral
            @jit
            def integrate_global(arr: np.ndarray) -> NumberOrArray:
                """integrate data

                Args:
                    arr (:class:`~numpy.ndarray`): discretized data on grid
                """
                return integrate_local(arr)

        else:
            # we are in a parallel run, so we need to gather the sub-integrals from all
            # subgrids in the grid mesh
            from ..tools.mpi import mpi_allreduce

            @jit
            def integrate_global(arr: np.ndarray) -> NumberOrArray:
                """integrate data over MPI parallelized grid

                Args:
                    arr (:class:`~numpy.ndarray`): discretized data on grid
                """
                integral = integrate_local(arr)
                return mpi_allreduce(integral)  # type: ignore

        return integrate_global  # type: ignore


def registered_operators() -> dict[str, list[str]]:
    """returns all operators that are currently defined

    Returns:
        dict: a dictionary with the names of the operators defined for each grid class
    """
    return {
        name: sorted(cls.operators)
        for name, cls in GridBase._subclasses.items()
        if not (name.endswith("Base") or hasattr(cls, "deprecated") and cls.deprecated)
    }
