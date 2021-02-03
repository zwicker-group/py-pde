"""
Bases classes

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
"""

import functools
import inspect
import itertools
import json
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numba as nb
import numpy as np

from ..tools.cache import cached_method, cached_property
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, classproperty
from ..tools.numba import jit

if TYPE_CHECKING:
    from .boundaries.axes import Boundaries, BoundariesData  # @UnusedImport


ArrayLike = Union[np.ndarray, Number]
OptionalArrayLike = Optional[ArrayLike]

PI_4 = 4 * np.pi
PI_43 = 4 / 3 * np.pi


class Operator(NamedTuple):
    """ stores information about an operator """

    factory: Callable
    rank_in: int
    rank_out: int


def _check_shape(shape) -> Tuple[int, ...]:
    """ checks the consistency of shape tuples """
    if not hasattr(shape, "__iter__"):
        shape = [shape]  # support single numbers

    if len(shape) == 0:
        raise ValueError("Require at least one dimension")

    # convert the shape to a tuple of integers
    result = []
    for dim in shape:
        if dim == int(dim) and dim >= 1:
            result.append(int(dim))
        else:
            raise ValueError(f"{repr(dim)} is not a valid number of support points")
    return tuple(result)


def discretize_interval(
    x_min: float, x_max: float, num: int
) -> Tuple[np.ndarray, float]:
    r""" construct a list of equidistantly placed intervals 

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
    """ exception indicating that point lies outside domain """

    pass


class DimensionError(ValueError):
    """ exception indicating that dimensions were inconsistent """

    pass


class PeriodicityError(RuntimeError):
    """ exception indicating that the grid periodicity is inconsistent """

    pass


class GridBase(metaclass=ABCMeta):
    """ Base class for all grids defining common methods and interfaces """

    _subclasses: Dict[str, "GridBase"] = {}  # all classes inheriting from this
    _operators: Dict[str, Operator] = {}  # all operators defined for the grid

    # properties that are defined in subclasses
    dim: int  # int: The spatial dimension in which the grid is embedded
    axes: List[str]  # list: Name of all axes that are described by the grid
    axes_symmetric: List[str] = []
    """ list: The names of the additional axes that the fields do not depend on,
    e.g. along which they are constant. """

    cell_volume_data: Sequence[Union[float, np.ndarray]]
    coordinate_constraints: List[int] = []  # axes not described explicitly
    num_axes: int
    periodic: List[bool]

    # mandatory, immutable, private attributes
    _axes_bounds: Tuple[Tuple[float, float], ...]
    _axes_coords: Tuple[np.ndarray, ...]
    _discretization: np.array
    _shape: Tuple[int, ...]

    # to help sphinx, we here list docstrings for classproperties
    operators: Set[str]
    """ set: names of all operators defined for this grid """

    def __init__(self):
        """ initialize the grid """
        self._logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
        cls._operators: Dict[str, Callable] = {}

    @classmethod
    def from_state(cls, state: Union[str, Dict[str, Any]]) -> "GridBase":
        """create a field from a stored `state`.

        Args:
            state (`str` or `dict`):
                The state from which the grid is reconstructed. If `state` is a
                string, it is decoded as JSON, which should yield a `dict`.
        """
        # decode the json data
        if isinstance(state, str):
            state = dict(json.loads(state))

        # create the instance
        # create the instance of the correct class
        class_name = state.pop("class")
        if class_name == cls.__name__:
            raise RuntimeError(f"Cannot reconstruct abstract class `{class_name}`")
        grid_cls = cls._subclasses[class_name]
        return grid_cls.from_state(state)

    @property
    def axes_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """ tuple: lower and upper bounds of each axis """
        return self._axes_bounds

    @property
    def axes_coords(self) -> Tuple[np.ndarray, ...]:
        """ tuple: coordinates of the cells for each axis """
        return self._axes_coords

    @property
    def discretization(self) -> np.array:
        """ :class:`numpy.array`: the linear size of a cell along each axis """
        return self._discretization

    @property
    def shape(self) -> Tuple[int, ...]:
        """ tuple of int: the number of support points of each axis """
        return self._shape

    @abstractproperty
    def state(self) -> Dict[str, Any]:
        pass

    @property
    def state_serialized(self) -> str:
        """ str: JSON-serialized version of the state of this grid """
        state = self.state
        state["class"] = self.__class__.__name__
        return json.dumps(state)

    def copy(self) -> "GridBase":
        """ return a copy of the grid """
        return self.__class__.from_state(self.state)

    __copy__ = copy

    def __deepcopy__(self, memo: Dict[int, Any]) -> "GridBase":
        """create a deep copy of the grid. This function is for instance called when
        a grid instance appears in another object that is copied using `copy.deepcopy`
        """
        # this implementation assumes that a simple call to copy is sufficient
        result = self.copy()
        memo[id(self)] = result
        return result

    def __repr__(self):
        """ return instance as string """
        args = ", ".join(str(k) + "=" + str(v) for k, v in self.state.items())
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.axes_bounds == other.axes_bounds
            and self.periodic == other.periodic
        )

    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """
        return hash(
            (
                self.__class__.__name__,
                self.shape,
                self.axes_bounds,
                hash(tuple(self.periodic)),
            )
        )

    def compatible_with(self, other: "GridBase") -> bool:
        """tests whether this class is compatible with other grids.

        Grids are compatible when they cover the same area with the same
        discretization. The difference to equality is that compatible grids do
        not need to have the same periodicity in their boundaries.

        Args:
            other (:class:`~pde.grids.GridBase`):
                The other grid to test against

        Returns:
            bool: Whether the grid is compatible
        """
        return (
            self.__class__ == other.__class__
            and self.shape == other.shape
            and self.axes_bounds == other.axes_bounds
        )

    def assert_grid_compatible(self, other: "GridBase"):
        """checks whether `other` is compatible with the current grid

        Args:
            other (:class:`~pde.grids.GridBase`):
                The grid compared to this one

        Raises:
            ValueError: if grids are not compatible
        """
        if not self.compatible_with(other):
            raise ValueError(f"Grids {self} and {other} are incompatible")

    @property
    def numba_type(self) -> str:
        """ str: represents type of the grid data in numba signatures """
        return "f8[" + ", ".join([":"] * self.num_axes) + "]"

    @cached_property()
    def cell_coords(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: the coordinates of each cell """
        return np.moveaxis(np.meshgrid(*self.axes_coords, indexing="ij"), 0, -1)

    @cached_property()
    def cell_volumes(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: volume of each cell """
        vols = functools.reduce(np.outer, self.cell_volume_data)
        return np.broadcast_to(vols, self.shape)

    @cached_property()
    def uniform_cell_volumes(self) -> bool:
        """ bool: returns True if all cell volumes are the same """
        return all(np.asarray(vols).ndim == 0 for vols in self.cell_volume_data)

    def distance_real(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate the distance between two points given in real coordinates

        This takes periodic boundary conditions into account if need be

        Args:
            p1 (:class:`numpy.ndarray`): First position
            p2 (:class:`numpy.ndarray`): Second position

        Returns:
            float: Distance between the two positions
        """
        diff = self.difference_vector_real(p1, p2)
        return np.linalg.norm(diff, axis=-1)  # type: ignore

    def _iter_boundaries(self) -> Iterator[Tuple[int, bool]]:
        """iterate over all boundaries of the grid

        Yields:
            tuple: for each boundary, the generator returns a tuple indicating
            the axis of the boundary together with a boolean value indicating
            whether the boundary lies on the upper side of the axis.
        """
        return itertools.product(range(self.num_axes), [True, False])

    def _boundary_coordinates(self, axis: int, upper: bool) -> np.ndarray:
        """get indices for accessing the points on the boundary

        Args:
            axis (int):
                The axis perpendicular to the boundary
            upper (bool):
                Whether the boundary is at the upper side of the axis

        Returns:
            :class:`numpy.ndarray`: Coordinates of the boundary points.
        """
        # get coordinate along the axis determining the boundary
        if upper:
            c_bndry = np.array([self._axes_bounds[axis][1]])
        else:
            c_bndry = np.array([self._axes_bounds[axis][0]])

        # get orthogonal coordinates
        coords = tuple(
            c_bndry if i == axis else self._axes_coords[i] for i in range(self.num_axes)
        )
        points = np.meshgrid(*coords, indexing="ij")

        # assemble into array
        shape_bndry = tuple(self.shape[i] for i in range(self.num_axes) if i != axis)
        shape = shape_bndry + (self.num_axes,)
        return np.stack(points, -1).reshape(shape)

    @abstractproperty
    def volume(self) -> float:
        pass

    @abstractmethod
    def cell_to_point(self, cells: np.ndarray, cartesian: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def point_to_cell(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def point_to_cartesian(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def point_from_cartesian(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def difference_vector_real(self, p1: np.ndarray, p2: np.ndarray):
        pass

    @abstractmethod
    def polar_coordinates_real(self, origin: np.ndarray, *, ret_angle: bool = False):
        pass

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def iter_mirror_points(
        self, point: np.ndarray, with_self: bool = False, only_periodic: bool = True
    ) -> Generator:
        pass

    @abstractmethod
    def get_boundary_conditions(
        self, bc: "BoundariesData" = "natural", rank: int = 0
    ) -> "Boundaries":
        pass

    @abstractmethod
    def get_line_data(self, data: np.ndarray, extract: str = "auto") -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_image_data(self, data: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_random_point(
        self, boundary_distance: float = 0, cartesian: bool = True
    ) -> np.ndarray:
        pass

    def normalize_point(self, point: np.ndarray, reflect: bool = True) -> np.ndarray:
        """normalize coordinates by applying periodic boundary conditions

        Args:
            point (:class:`numpy.ndarray`):
                Coordinates of a single point
            reflect (bool):
                Flag determining whether coordinates along non-periodic axes are
                reflected to lie in the valid range. If `False`, such coordinates are
                left unchanged and only periodic boundary conditions are enforced.

        Returns:
            :class:`numpy.ndarray`: The respective coordinates with periodic
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

    @classmethod
    def register_operator(
        cls,
        name: str,
        factory_func: Callable = None,
        rank_in: int = 0,
        rank_out: int = 0,
    ):
        """register an operator for this grid

        Example:
            The method can either be used directly::

                GridClass.register_operator("operator", make_operator)

            or as a decorator for the factory function::

                @GridClass.register_operator("operator")
                def make_operator(bcs: Boundaries):
                    ...

        Args:
            name (str):
                The name of the operator to register
            factory_func (callable):
                A function with signature ``(bcs: Boundaries, **kwargs)``, which
                takes boundary conditions and optional keyword arguments and
                returns an implementation of the given operator. This
                implementation is a function that takes a
                :class:`~numpy.ndarray` of discretized values as arguments and
                returns the resulting discretized data in a
                :class:`~numpy.ndarray` after applying the operator.
            rank_in (int):
                The rank of the input field for the operator
            rank_out (int):
                The rank of the field that is returned by the operator
        """

        def register_operator(factor_func_arg: Callable):
            """ helper function to register the operator """
            cls._operators[name] = Operator(
                factory=factor_func_arg, rank_in=rank_in, rank_out=rank_out
            )
            return factor_func_arg

        if factory_func is None:
            # method is used as a decorator, so return the helper function
            return register_operator
        else:
            # method is used directly
            register_operator(factory_func)

    @classproperty  # type: ignore
    def operators(cls) -> Set[str]:  # @NoSelf
        """ set: all operators defined for this class """
        result = set()
        classes = inspect.getmro(cls)[:-1]  # type: ignore
        for anycls in classes:
            result |= set(anycls._operators.keys())  # type: ignore
        return result

    @cached_method()
    @fill_in_docstring
    def get_operator(self, name: str, bc: "Boundaries", **kwargs) -> Callable:
        """return a discretized operator defined on this grid

        Args:
            name (str):
                Identifier for the operator. Some examples are 'laplace',
                'gradient', or 'divergence'. The registered operators for this
                grid can be obtained from the
                :attr:`~pde.grids.base.GridBase.operators` attribute.
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            **kwargs:
                Specifies extra arguments that can influence how the operator
                is created. Many operators support a `method` argument that can
                typically be set to 'numba', 'scipy', or `auto`.

        Returns:
            A function that takes the discretized data as an input and returns
            the data to which the operator `name` has been applied. This
            function optionally supports a second argument, which provides
            allocated memory for the output.
        """
        # obtain all parent classes, except `object`
        classes = inspect.getmro(self.__class__)[:-1]
        for cls in classes:
            if name in cls._operators:  # type: ignore
                operator = cls._operators[name]  # type: ignore
                rank = min(operator.rank_in, operator.rank_out)
                bcs = self.get_boundary_conditions(bc, rank=rank)
                return operator.factory(bcs, **kwargs)  # type: ignore

        # operator was not found
        op_list = ", ".join(sorted(self.operators))
        raise ValueError(
            f"'{name}' is not one of the defined operators ({op_list}). Custom "
            "operators can be added using the `register_operator` method."
        )

    def get_subgrid(self, indices: Sequence[int]) -> "GridBase":
        """ return a subgrid of only the specified axes """
        raise NotImplementedError(
            f"Subgrids are not implemented for class {self.__class__.__name__}"
        )

    def plot(self):
        """ visualize the grid """
        raise NotImplementedError(
            f"Plotting is not implemented for class {self.__class__.__name__}"
        )

    @property
    def typical_discretization(self) -> float:
        """ float: the average side length of the cells """
        return np.mean(self.discretization)  # type: ignore

    def integrate(
        self, data: np.ndarray, axes: Union[int, Sequence[int]] = None
    ) -> np.ndarray:
        """Integrates the discretized data over the grid

        Args:
            data (:class:`numpy.ndarray`):
                The values at the support points of the grid that need to be
                integrated.
            axes (list of int, optional):
                The axes along which the integral is performed. If omitted, all
                axes are integrated over.

        Returns:
            :class:`numpy.ndarray`: The values integrated over the entire grid
        """
        # determine the volumes of the individual cells
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
        return (data * cell_volumes).sum(axis=axes)

    @cached_method()
    def make_normalize_point_compiled(self, reflect: bool = True) -> Callable:
        """return a compiled function that normalizes the points

        Normalizing points is useful to respect periodic boundary conditions.
        Here, points are assumed to be specified by the physical values along
        the non-symmetric axes of the grid.

        Args:
            reflect (bool):
                Flag determining whether coordinates along non-periodic axes are
                reflected to lie in the valid range. If `False`, such coordinates are
                left unchanged and only periodic boundary conditions are enforced.

        Returns:
            callable: A function that takes a :class:`numpy.ndarray` as an argument,
            which describes the coordinates of the points. This array is modified
            in-place!
        """
        num_axes = self.num_axes
        periodic = tuple(self.periodic)
        bounds = np.array(self.axes_bounds)
        xmin = bounds[:, 0]
        xmax = bounds[:, 1]
        size = bounds[:, 1] - bounds[:, 0]

        @jit
        def normalize_point(point: np.ndarray) -> np.ndarray:
            """ helper function normalizing a single point """
            for i in range(num_axes):
                if periodic[i]:
                    point[i] = (point[i] - xmin[i]) % size[i] + xmin[i]
                elif reflect:
                    arg = (point[i] - xmax[i]) % (2 * size[i]) - size[i]
                    point[i] = xmin[i] + abs(arg)

        return normalize_point  # type: ignore

    @cached_method()
    def make_cell_volume_compiled(self, flat_index: bool = False) -> Callable:
        """return a compiled function returning the volume of a grid cell

        Args:
            flat_index (bool):
                When True, cell_volumes are indexed by a single integer into the
                flattened array.

        Returns:
            function: returning the volume of the chosen cell
        """
        if all(np.isscalar(d) for d in self.cell_volume_data):
            # all cells have the same volume
            cell_volume = np.product(self.cell_volume_data)

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

    @fill_in_docstring
    def make_interpolator_compiled(
        self, bc: "BoundariesData" = "natural", rank: int = 0, fill: Number = None
    ) -> Callable:
        """return a compiled function for linear interpolation on the grid

        This interpolator respects boundary conditions and can thus interpolate
        values in the whole grid volume. However, close to corners, the
        interpolation might not be optimal, in particular for periodic grids.

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            rank (int, optional):
                The tensorial rank of the value associated with the boundary
                condition.
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.

        Returns:
            A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. The signature of
            this function is (data, point), where `data` is the numpy array
            containing the field data and position is denotes the position in
            grid coordinates.
        """
        bcs = self.get_boundary_conditions(bc, rank=rank)

        if self.num_axes == 1:
            # specialize for 1-dimensional interpolation
            size = self.shape[0]
            lo = self.axes_bounds[0][0]
            dx = self.discretization[0]
            periodic = self.periodic[0]
            ev = bcs[0].get_point_evaluator(fill=fill)

            @jit
            def interpolate_single(data: np.ndarray, point: np.ndarray) -> np.ndarray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`numpy.ndarray`):
                        A 1d array of values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system

                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                c_l, d_l = divmod((point[0] - lo) / dx - 0.5, 1.0)
                if periodic:
                    c_li = int(c_l) % size
                    c_hi = (c_li + 1) % size
                else:
                    if c_l < -1 or c_l > size - 1:
                        if fill is None:
                            raise DomainError("Point lies outside the grid")
                        else:
                            return fill
                    c_li = int(c_l)
                    c_hi = c_li + 1
                return (1 - d_l) * ev(data, (c_li,)) + d_l * ev(data, (c_hi,))

        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            size_x, size_y = self.shape
            lo_x, lo_y = np.array(self.axes_bounds)[:, 0]
            dx, dy = self.discretization
            periodic_x, periodic_y = self.periodic
            ev_x = bcs[0].get_point_evaluator(fill=fill)
            ev_y = bcs[1].get_point_evaluator(fill=fill)

            @jit
            def interpolate_single(data: np.ndarray, point: np.ndarray) -> np.ndarray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`numpy.ndarray`):
                        The values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system

                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.0)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.0)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)

                value = np.zeros(data.shape[:-2], dtype=data.dtype)
                weight = 0
                for i in range(2):
                    c_x = int(c_lx) + i
                    if periodic_x:
                        c_x %= size_x
                        inside_x = True
                    else:
                        inside_x = -1 < c_x < size_x

                    for j in range(2):
                        c_y = int(c_ly) + j
                        if periodic_y:
                            c_y %= size_y
                            inside_y = True
                        else:
                            inside_y = -1 < c_y < size_y

                        w = w_x[i] * w_y[j]
                        if inside_x and inside_y:
                            value += w * data[..., c_x, c_y]
                            weight += w
                        elif not inside_x and inside_y:
                            value += w * ev_x(data, (c_x, c_y))
                            weight += w
                        elif inside_x and not inside_y:
                            value += w * ev_y(data, (c_x, c_y))
                            weight += w
                        # else: ignore points that are not inside any of the
                        # axes, where we would have to do interpolation along
                        # two axes. This would in principle be possible for
                        # periodic boundary conditions, but this is tedious to
                        # implement correctly.

                if weight == 0:
                    if fill is None:
                        raise DomainError("Point lies outside the grid")
                    else:
                        return fill

                return value / weight

        elif self.num_axes == 3:
            # specialize for 3-dimensional interpolation
            size_x, size_y, size_z = self.shape
            lo_x, lo_y, lo_z = np.array(self.axes_bounds)[:, 0]
            dx, dy, dz = self.discretization
            periodic_x, periodic_y, periodic_z = self.periodic
            ev_x = bcs[0].get_point_evaluator(fill=fill)
            ev_y = bcs[1].get_point_evaluator(fill=fill)
            ev_z = bcs[2].get_point_evaluator(fill=fill)

            @jit
            def interpolate_single(data: np.ndarray, point: np.ndarray) -> np.ndarray:
                """obtain interpolated value of data at a point

                Args:
                    data (:class:`numpy.ndarray`):
                        The values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system

                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.0)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.0)
                c_lz, d_lz = divmod((point[2] - lo_z) / dz - 0.5, 1.0)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)
                w_z = (1 - d_lz, d_lz)

                value = np.zeros(data.shape[:-3], dtype=data.dtype)
                weight = 0
                for i in range(2):
                    c_x = int(c_lx) + i
                    if periodic_x:
                        c_x %= size_x
                        inside_x = True
                    else:
                        inside_x = -1 < c_x < size_x

                    for j in range(2):
                        c_y = int(c_ly) + j
                        if periodic_y:
                            c_y %= size_y
                            inside_y = True
                        else:
                            inside_y = -1 < c_y < size_y

                        for k in range(2):
                            c_z = int(c_lz) + k
                            if periodic_z:
                                c_z %= size_z
                                inside_z = True
                            else:
                                inside_z = -1 < c_z < size_z

                            w = w_x[i] * w_y[j] * w_z[k]
                            if inside_x and inside_y and inside_z:
                                value += w * data[..., c_x, c_y, c_z]
                                weight += w
                            elif not inside_x and inside_y and inside_z:
                                value += w * ev_x(data, (c_x, c_y, c_z))
                                weight += w
                            elif inside_x and not inside_y and inside_z:
                                value += w * ev_y(data, (c_x, c_y, c_z))
                                weight += w
                            elif inside_x and inside_y and not inside_z:
                                value += w * ev_z(data, (c_x, c_y, c_z))
                                weight += w
                            # else: ignore points that would need to be
                            # interpolated along more than one axis.
                            # Implementing this would in principle be possible
                            # for periodic boundary conditions, but this is
                            # tedious to do correctly.

                if weight == 0:
                    if fill is None:
                        raise DomainError("Point lies outside the grid")
                    else:
                        return fill

                return value / weight

        else:
            raise NotImplementedError(
                f"Compiled interpolation not implemented for dimension {self.num_axes}"
            )

        return interpolate_single  # type: ignore

    def make_add_interpolated_compiled(self) -> Callable:
        """return a compiled function to add amounts at interpolated positions

        Returns:
            A function with signature (data, position, amount), where `data` is
            the numpy array containing the field data, position is denotes the
            position in grid coordinates, and `amount` is the  that is to be
            added to the field.
        """
        cell_volume = self.make_cell_volume_compiled()

        if self.num_axes == 1:
            # specialize for 1-dimensional interpolation
            lo = self.axes_bounds[0][0]
            dx = self.discretization[0]
            size = self.shape[0]
            periodic = bool(self.periodic[0])

            @jit
            def add_interpolated(data: np.ndarray, point: np.ndarray, amount):
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`numpy.ndarray`):
                        The values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system
                    amount (float or :class:`numpy.ndarray`):
                        The amount that will be added to the data. This value
                        describes an integrated quantity (given by the field
                        value times the discretization volume). This is
                        important for consistency with different discretizations
                        and in particular grids with non-uniform discretizations
                """
                c_l, d_l = divmod((point[0] - lo) / dx - 0.5, 1.0)
                if c_l < -1 or c_l > size - 1:
                    raise DomainError("Point lies outside grid")
                c_li = int(c_l)
                c_hi = c_li + 1

                if periodic:
                    c_li %= size
                    c_hi %= size
                    w_l = 1 - d_l  # weights of the low point
                    w_h = d_l  # weights of the high point
                    data[..., c_li] += w_l * amount / cell_volume(c_li)
                    data[..., c_hi] += w_h * amount / cell_volume(c_hi)

                elif c_li < 0:
                    if c_hi >= size:
                        raise DomainError("Point lies outside the grid")
                    else:  # c_hi < size
                        data[..., c_hi] += amount / cell_volume(c_hi)
                else:  # c_li >= 0
                    if c_hi >= size:
                        data[..., c_li] += amount / cell_volume(c_li)
                    else:  # c_hi < size
                        w_l = 1 - d_l  # weights of the low point
                        w_h = d_l  # weights of the high point
                        data[..., c_li] += w_l * amount / cell_volume(c_li)
                        data[..., c_hi] += w_h * amount / cell_volume(c_hi)

        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            size_x, size_y = self.shape
            lo_x, lo_y = np.array(self.axes_bounds)[:, 0]
            dx, dy = self.discretization
            periodic_x, periodic_y = self.periodic

            @jit
            def add_interpolated(data: np.ndarray, point: np.ndarray, amount):
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`numpy.ndarray`):
                        The values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system
                    amount (Number or :class:`numpy.ndarray`):
                        The amount that will be added to the data. This value
                        describes an integrated quantity (given by the field
                        value times the discretization volume). This is
                        important for consistency with different discretizations
                        and in particular grids with non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.0)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.0)
                c_xi = int(c_lx)
                c_yi = int(c_ly)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)

                # determine the total weight
                total_weight = 0
                for i in range(2):
                    c_x = c_xi + i
                    if periodic_x:
                        c_x %= size_x
                    elif not (0 <= c_x < size_x):  # inside x?
                        continue
                    for j in range(2):
                        c_y = c_yi + j
                        if periodic_y:
                            c_y %= size_y
                        elif not (0 <= c_y < size_y):  # inside y?
                            continue
                        total_weight += w_x[i] * w_y[j]

                if total_weight == 0:
                    raise DomainError("Point lies outside the grid")

                # change the field with the correct weights
                for i in range(2):
                    c_x = c_xi + i
                    if periodic_x:
                        c_x %= size_x
                    elif not (0 <= c_x < size_x):  # inside x?
                        continue
                    for j in range(2):
                        c_y = c_yi + j
                        if periodic_y:
                            c_y %= size_y
                        elif not (0 <= c_y < size_y):  # inside y?
                            continue
                        w = w_x[i] * w_y[j] / total_weight
                        cell_vol = cell_volume(c_x, c_y)
                        data[..., c_x, c_y] += w * amount / cell_vol

        elif self.num_axes == 3:
            # specialize for 3-dimensional interpolation
            size_x, size_y, size_z = self.shape
            lo_x, lo_y, lo_z = np.array(self.axes_bounds)[:, 0]
            dx, dy, dz = self.discretization
            periodic_x, periodic_y, periodic_z = self.periodic

            @jit
            def add_interpolated(data: np.ndarray, point: np.ndarray, amount):
                """add an amount to a field at an interpolated position

                Args:
                    data (:class:`numpy.ndarray`):
                        The values at the grid points
                    point (:class:`numpy.ndarray`):
                        Coordinates of a single point in the grid coordinate
                        system
                    amount (Number or :class:`numpy.ndarray`):
                        The amount that will be added to the data. This value
                        describes an integrated quantity (given by the field
                        value times the discretization volume). This is
                        important for consistency with different discretizations
                        and in particular grids with non-uniform discretizations
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.0)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.0)
                c_lz, d_lz = divmod((point[2] - lo_z) / dz - 0.5, 1.0)
                c_xi = int(c_lx)
                c_yi = int(c_ly)
                c_zi = int(c_lz)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)
                w_z = (1 - d_lz, d_lz)

                # determine the total weight
                total_weight = 0
                for i in range(2):
                    c_x = c_xi + i
                    if periodic_x:
                        c_x %= size_x
                    elif not (0 <= c_x < size_x):  # inside x?
                        continue

                    for j in range(2):
                        c_y = c_yi + j
                        if periodic_y:
                            c_y %= size_y
                        elif not (0 <= c_y < size_y):  # inside y?
                            continue

                        for k in range(2):
                            c_z = c_zi + k
                            if periodic_z:
                                c_z %= size_z
                            elif not (0 <= c_z < size_z):  # inside z?
                                continue

                            # only consider the points inside the grid
                            total_weight += w_x[i] * w_y[j] * w_z[k]

                if total_weight == 0:
                    raise DomainError("Point lies outside the grid")

                # change the field with the correct weights
                for i in range(2):
                    c_x = c_xi + i
                    if periodic_x:
                        c_x %= size_x
                    elif not (0 <= c_x < size_x):  # inside x?
                        continue

                    for j in range(2):
                        c_y = c_yi + j
                        if periodic_y:
                            c_y %= size_y
                        elif not (0 <= c_y < size_y):  # inside y?
                            continue

                        for k in range(2):
                            c_z = c_zi + k
                            if periodic_z:
                                c_z %= size_z
                            elif not (0 <= c_z < size_z):  # inside z?
                                continue

                            w = w_x[i] * w_y[j] * w_z[k] / total_weight
                            cell_vol = cell_volume(c_x, c_y, c_z)
                            data[..., c_x, c_y, c_z] += w * amount / cell_vol

        else:
            raise NotImplementedError(
                f"Compiled interpolation not implemented for dimension {self.num_axes}"
            )

        return add_interpolated  # type: ignore

    def make_integrator(self) -> Callable:
        """Return function that can be used to integrates discretized data over the grid

        Note that currently only scalar fields are supported.

        Returns:
            callable: A function that takes a numpy array and returns the integral with
            the correct weights given by the cell volumes.
        """
        num_axes = self.num_axes

        if self.uniform_cell_volumes:
            # all cells have the same volume
            cell_volume = np.product(self.cell_volume_data)

            @jit
            def integrate(arr: np.ndarray) -> ArrayLike:
                """ function that integrates data over a uniform grid """
                assert arr.ndim == num_axes
                return cell_volume * arr.sum()

        else:
            # cell volume varies with position
            get_cell_volume = self.make_cell_volume_compiled(flat_index=True)

            @jit
            def integrate(arr: np.ndarray) -> float:
                """ function that integrates scalar data over a non-uniform grid """
                assert arr.ndim == num_axes
                total = 0
                for i in nb.prange(arr.size):
                    total += get_cell_volume(i) * arr.flat[i]
                return total

        return integrate  # type: ignore
