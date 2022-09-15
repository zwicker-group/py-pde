"""
Spherically-symmetric grids in 2 and 3 dimensions. These are grids that only
discretize the radial direction, assuming symmetry with respect to all angles.
This choice implies that differential operators might not be applicable to all fields.
For instance, the divergence of a vector field on a spherical grid can only be
represented as a scalar field on the same grid if the θ-component of the vector field
vanishes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
"""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Any, Dict, Generator, Tuple, TypeVar, Union

import numpy as np

from ..tools.cache import cached_property
from ..tools.plotting import plot_on_axes
from .base import DimensionError, GridBase, _check_shape, discretize_interval
from .cartesian import CartesianGrid

if TYPE_CHECKING:
    from .boundaries.axes import Boundaries  # @UnusedImport


π = np.pi
PI_4 = 4 * π
PI_43 = 4 / 3 * π


TNumArr = TypeVar("TNumArr", float, np.ndarray)


def volume_from_radius(radius: TNumArr, dim: int) -> TNumArr:
    """Return the volume of a sphere with a given radius

    Args:
        radius (float or :class:`~numpy.ndarray`): Radius of the sphere
        dim (int): Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Volume of the sphere
    """
    if dim == 1:
        return 2 * radius
    elif dim == 2:
        return π * radius**2
    elif dim == 3:
        return PI_43 * radius**3
    else:
        raise NotImplementedError(f"Cannot calculate the volume in {dim} dimensions")


class SphericalSymGridBase(GridBase, metaclass=ABCMeta):  # lgtm [py/missing-equals]
    r"""Base class for d-dimensional spherical grids with angular symmetry

    The angular symmetry implies that states only depend on the radial
    coordinate :math:`r`, which is discretized uniformly as

    .. math::
        r_i = R_\mathrm{inner} + \left(i + \frac12\right) \Delta r
        \quad \text{for} \quad i = 0, \ldots, N - 1
        \quad \text{with} \quad
            \Delta r = \frac{R_\mathrm{outer} - R_\mathrm{inner}}{N}

    where :math:`R_\mathrm{outer}` is the outer radius of the grid and
    :math:`R_\mathrm{inner}` corresponds to a possible inner radius, which is
    zero by default. The radial direction is discretized by :math:`N` support
    points.
    """

    periodic = [False]  # the radial axis is not periodic
    num_axes = 1  # the number of independent axes

    boundary_names = {"inner": (0, False), "outer": (0, True)}

    def __init__(
        self, radius: Union[float, Tuple[float, float]], shape: Union[Tuple[int], int]
    ):
        r"""
        Args:
            radius (float or tuple of floats):
                radius :math:`R_\mathrm{outer}` in case a simple float is given.
                If a tuple is supplied it is interpreted as the inner and outer
                radius, :math:`(R_\mathrm{inner}, R_\mathrm{outer})`.
            shape (tuple or int): A single number setting the number :math:`N`
                of support points along the radial coordinate
        """
        super().__init__()
        shape_list = _check_shape(shape)
        if not len(shape_list) == 1:
            raise ValueError(f"`shape` must be a single number, not {shape_list}")
        self._shape: Tuple[int] = (int(shape_list[0]),)

        try:
            r_inner, r_outer = radius  # type: ignore
        except TypeError:
            r_inner, r_outer = 0, float(radius)  # type: ignore

        if r_inner < 0:
            raise ValueError("Inner radius must be positive")
        if r_inner >= r_outer:
            raise ValueError("Outer radius must be larger than inner radius")

        # radial discretization
        rs, dr = discretize_interval(r_inner, r_outer, self.shape[0])

        self._axes_coords = (rs,)
        self._axes_bounds = ((r_inner, r_outer),)
        self._discretization = np.array((dr,))

    @property
    def state(self) -> Dict[str, Any]:
        """state: the state of the grid"""
        return {"radius": self.radius, "shape": self.shape}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> SphericalSymGridBase:  # type: ignore
        """create a field from a stored `state`.

        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(radius=state_copy.pop("radius"), shape=state_copy.pop("shape"))
        if state_copy:
            raise ValueError(f"State items {state_copy.keys()} were not used")
        return obj

    @classmethod
    def from_bounds(  # type: ignore
        cls,
        bounds: Tuple[Tuple[float, float]],
        shape: Tuple[int],
        periodic: Tuple[bool],
    ) -> SphericalSymGridBase:
        """
        Args:
            bounds (tuple): Give the coordinate range for the radial axis.
            shape (tuple): The number of support points for the radial axis
            periodic (bool or list): Not used

        Returns:
            SphericalGridBase representing the region chosen by bounds
        """
        if len(bounds) != 1:
            raise ValueError(
                f"`bounds` must be given as ((r_min, r_max),). Got {bounds} instead"
            )
        return cls(bounds[0], shape)

    @property
    def has_hole(self) -> bool:
        """returns whether the inner radius is larger than zero"""
        return self.axes_bounds[0][0] > 0

    @property
    def radius(self) -> Union[float, Tuple[float, float]]:
        """float: radius of the sphere"""
        r_inner, r_outer = self.axes_bounds[0]
        if r_inner == 0:
            return r_outer
        else:
            return r_inner, r_outer

    @property
    def volume(self) -> float:
        """float: total volume of the grid"""
        r_inner, r_outer = self.axes_bounds[0]
        volume = volume_from_radius(r_outer, dim=self.dim)
        if r_inner > 0:
            volume -= volume_from_radius(r_inner, dim=self.dim)
        return volume

    @cached_property()
    def cell_volume_data(self) -> Tuple[np.ndarray]:
        """tuple of :class:`~numpy.ndarray`: the volumes of all cells"""
        dr = self.discretization[0]
        rs = self.axes_coords[0]
        volumes_h = volume_from_radius(rs + 0.5 * dr, dim=self.dim)
        volumes_l = volume_from_radius(rs - 0.5 * dr, dim=self.dim)
        return ((volumes_h - volumes_l).reshape(self.shape[0]),)

    def get_random_point(
        self,
        *,
        boundary_distance: float = 0,
        avoid_center: bool = False,
        coords: str = "cartesian",
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """return a random point within the grid

        Note that these points will be uniformly distributed in the volume, implying
        they are not uniformly distributed on the radial axis.

        Args:
            boundary_distance (float):
                The minimal distance this point needs to have from all boundaries.
            avoid_center (bool):
                Determines whether the boundary distance should also be kept from the
                center, i.e., whether points close to the center are returned.
            coords (str):
                Determines the coordinate system in which the point is specified. Valid
                values are `cartesian`, `cell`, and `grid`;
                see :meth:`~pde.grids.base.GridBase.transform`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)

        Returns:
            :class:`~numpy.ndarray`: The coordinates of the point
        """
        if rng is None:
            rng = np.random.default_rng()

        # handle the boundary distance
        r_inner, r_outer = self.axes_bounds[0]
        r_min = r_inner + boundary_distance if avoid_center else r_inner
        r_max = r_outer - boundary_distance
        if r_max <= r_min:
            raise RuntimeError("Random points would be too close to boundary")

        # choose random radius scaled such that points are uniformly distributed
        r = np.array(
            [rng.uniform(r_min**self.dim, r_max**self.dim) ** (1 / self.dim)]
        )
        if coords == "cartesian":
            # choose random angles for the already chosen radius
            if self.dim == 2:
                φ = rng.uniform(0, 2 * np.pi)
                point = np.r_[r, φ]
            elif self.dim == 3:
                θ = np.arccos(rng.uniform(-1, 1))
                φ = rng.uniform(0, 2 * np.pi)
                point = np.r_[r, θ, φ]
            else:
                raise NotImplementedError(f"{self.dim} dimensions")

            return self.point_to_cartesian(point, full=True)

        elif coords == "cell":
            return self.transform(r, "grid", "cell")

        elif coords == "grid":
            return r

        else:
            raise ValueError(f"Unknown coordinate system `{coords}`")

    def get_line_data(self, data: np.ndarray, extract: str = "auto") -> Dict[str, Any]:
        """return a line cut along the radial axis

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points
            extract (str):
                Determines which cut is done through the grid. This parameter is
                mainly supplied for a consistent interface and has no effect for
                polar grids.

        Returns:
            A dictionary with information about the line cut, which is
            convenient for plotting.
        """
        if extract not in {"auto", "r", "radial"}:
            raise ValueError(f"Unknown extraction method `{extract}`")

        return {
            "data_x": self.axes_coords[0],
            "data_y": data,
            "extent_x": self.axes_bounds[0],
            "label_x": self.axes[0],
        }

    def get_image_data(
        self,
        data: np.ndarray,
        performance_goal: str = "speed",
        fill_value: float = 0,
        masked: bool = True,
    ) -> Dict[str, Any]:
        """return a 2d-image of the data

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points
            performance_goal (str):
                Determines the method chosen for interpolation. Possible options
                are `speed` and `quality`.
            fill_value (float):
                The value assigned to invalid positions (those inside the hole
                or outside the region).
            masked (bool):
                Whether a :class:`numpy.ma.MaskedArray` is returned for the data
                instead of the normal :class:`~numpy.ndarray`.

        Returns:
            A dictionary with information about the image, which is  convenient
            for plotting.
        """
        from scipy import interpolate

        _, r_outer = self.axes_bounds[0]
        r_data = self.axes_coords[0]

        if self.has_hole:
            num = int(np.ceil(r_outer / self.discretization[0]))
            x_positive, _ = discretize_interval(0, r_outer, num)
        else:
            x_positive = r_data

        x = np.r_[-x_positive[::-1], x_positive]
        xs, ys = np.meshgrid(x, x, indexing="ij")
        r_img = np.hypot(xs, ys)

        if performance_goal == "speed":
            # interpolate over the new coordinates using linear interpolation
            f = interpolate.interp1d(
                r_data,
                data,
                copy=False,
                bounds_error=False,
                fill_value=fill_value,
                assume_sorted=True,
            )
            data_int = f(r_img.flat).reshape(r_img.shape)

        elif performance_goal == "quality":
            # interpolate over the new coordinates using radial base function
            f = interpolate.Rbf(r_data, data, function="cubic")
            data_int = f(r_img)

        else:
            raise ValueError(f"Performance goal `{performance_goal}` undefined")

        if masked:
            mask = (r_img < r_data[0]) | (r_data[-1] < r_img)
            data_int = np.ma.masked_array(data_int, mask=mask)

        return {
            "data": data_int,
            "x": x,
            "y": x,
            "extent": (-r_outer, r_outer, -r_outer, r_outer),
            "label_x": "x",
            "label_y": "y",
        }

    def iter_mirror_points(
        self, point: np.ndarray, with_self: bool = False, only_periodic: bool = True
    ) -> Generator:
        """generates all mirror points corresponding to `point`

        Args:
            point (:class:`~numpy.ndarray`): the point within the grid
            with_self (bool): whether to include the point itself
            only_periodic (bool): whether to only mirror along periodic axes

        Returns:
            A generator yielding the coordinates that correspond to mirrors
        """
        if with_self:
            yield np.asanyarray(point, dtype=np.double)

    def point_from_cartesian(self, points: np.ndarray) -> np.ndarray:
        """convert points given in Cartesian coordinates to this grid

        Args:
            points (:class:`~numpy.ndarray`):
                Points given in Cartesian coordinates.

        Returns:
            :class:`~numpy.ndarray`: Points given in the coordinates of the grid
        """
        points = np.atleast_1d(points)
        assert points.shape[-1] == self.dim, f"Point must have {self.dim} coordinates"
        return np.linalg.norm(points, axis=-1, keepdims=True)  # type: ignore

    def polar_coordinates_real(
        self, origin=None, *, ret_angle: bool = False, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """return spherical coordinates associated with the grid

        Args:
            origin:
                Place holder variable to comply with the interface
            ret_angle (bool):
                Determines whether angles are returned alongside the distance. If
                `False` only the distance to the origin is returned for each support
                point of the grid. If `True`, the distance and angles are returned. Note
                that in the case of spherical grids, this angle is zero by convention.
        """
        # check the consistency of the origin argument, which can be set for other grids
        if origin is not None:
            origin = np.array(origin, dtype=np.double, ndmin=1)
            if not np.array_equal(origin, np.zeros(self.dim)):
                raise RuntimeError(f"Origin must be {str([0]*self.dim)}")

        # the distance to the origin is exactly the radial coordinate
        rs = self.axes_coords[0]
        if ret_angle:
            return (rs,) + (np.zeros_like(rs),) * (self.dim - 1)
        else:
            return rs

    def get_cartesian_grid(self, mode: str = "valid", num: int = None) -> CartesianGrid:
        """return a Cartesian grid for this spherical one

        Args:
            mode (str):
                Determines how the grid is determined. Setting it to 'valid' (or
                'inscribed') only returns points that are fully resolved in the
                spherical grid, e.g., the Cartesian grid is inscribed in the sphere.
                Conversely, 'full' (or 'circumscribed') returns all data, so the
                Cartesian grid is circumscribed.
            num (int):
                Number of support points along each axis of the returned grid.

        Returns:
            :class:`pde.grids.cartesian.CartesianGrid`: The requested grid
        """
        # pick how the grid is determined
        if mode == "valid":
            if self.has_hole:
                self._logger.warning("Sphere has holes, so not all points are valid")
            bounds = self.radius / np.sqrt(self.dim)
        elif mode == "inscribed":
            bounds = self.radius / np.sqrt(self.dim)
        elif mode == "full" or mode == "circumscribed":
            bounds = self.radius
        else:
            raise ValueError(f"Unsupported mode `{mode}`")

        # determine the grid points
        if num is None:
            num = 2 * round(bounds / self.discretization[0])
        grid_bounds = [(-bounds, bounds)] * self.dim
        return CartesianGrid(grid_bounds, num)

    @plot_on_axes()
    def plot(self, ax, **kwargs):
        r"""visualize the spherically symmetric grid in two dimensions

        Args:
            {PLOT_ARGS}
            \**kwargs: Extra arguments are passed on the to the matplotlib
                plotting routines, e.g., to set the color of the lines
        """
        from matplotlib import collections, patches

        kwargs.setdefault("edgecolor", kwargs.get("color", "k"))
        kwargs.setdefault("facecolor", "none")
        (rb,) = self.axes_bounds
        rmax = rb[1]

        # draw circular parts
        circles = []
        for r in np.linspace(*rb, self.shape[0] + 1):
            if r == 0:
                c = patches.Circle((0, 0), 0.01 * rmax)
            else:
                c = patches.Circle((0, 0), r)
            circles.append(c)
        ax.add_collection(collections.PatchCollection(circles, **kwargs))

        ax.set_xlim(-rmax, rmax)
        ax.set_xlabel("x")
        ax.set_ylim(-rmax, rmax)
        ax.set_ylabel("y")
        ax.set_aspect(1)


class PolarSymGrid(SphericalSymGridBase):
    r"""2-dimensional polar grid assuming angular symmetry

    The angular symmetry implies that states only depend on the radial
    coordinate :math:`r`, which is discretized uniformly as

    .. math::
        r_i = R_\mathrm{inner} + \left(i + \frac12\right) \Delta r
        \quad \text{for} \quad i = 0, \ldots, N - 1
        \quad \text{with} \quad
            \Delta r = \frac{R_\mathrm{outer} - R_\mathrm{inner}}{N}

    where :math:`R_\mathrm{outer}` is the outer radius of the grid and
    :math:`R_\mathrm{inner}` corresponds to a possible inner radius, which is
    zero by default. The radial direction is discretized by :math:`N` support
    points.
    """

    dim = 2  # dimension of the described space
    axes = ["r"]
    axes_symmetric = ["phi"]
    coordinate_constraints = [0, 1]  # axes not described explicitly

    def point_to_cartesian(
        self, points: np.ndarray, *, full: bool = False
    ) -> np.ndarray:
        """convert coordinates of a point to Cartesian coordinates

        This function returns points along the y-coordinate, i.e, the x
        coordinates will be zero.

        Args:
            points (:class:`~numpy.ndarray`): The grid coordinates of the points
            full (bool): Flag indicating whether angular coordinates are specified

        Returns:
            :class:`~numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)

        if full:
            # angles are supplied
            if points.shape[-1] != self.dim:
                raise DimensionError(f"Shape {points.shape} cannot denote points")

            x = points[..., 0] * np.cos(points[..., 1])
            y = points[..., 0] * np.sin(points[..., 1])
        else:
            # angles are not supplied
            if points.shape[-1] != self.num_axes:
                raise DimensionError(f"Shape {points.shape} cannot denote points")

            y = points[..., 0]
            x = np.zeros_like(y)

        return np.stack((x, y), axis=-1)


class SphericalSymGrid(SphericalSymGridBase):
    r"""3-dimensional spherical grid assuming spherical symmetry

    The symmetry implies that states only depend on the radial coordinate
    :math:`r`, which is discretized as follows:

    .. math::
        r_i = R_\mathrm{inner} + \left(i + \frac12\right) \Delta r
        \quad \text{for} \quad i = 0, \ldots, N - 1
        \quad \text{with} \quad
            \Delta r = \frac{R_\mathrm{outer} - R_\mathrm{inner}}{N}

    where :math:`R_\mathrm{outer}` is the outer radius of the grid and
    :math:`R_\mathrm{inner}` corresponds to a possible inner radius, which is
    zero by default. The radial direction is discretized by :math:`N` support
    points.

    Warning:
        Not all results of differential operators on vectorial and tensorial fields can
        be expressed in terms of fields that only depend on the radial coordinate
        :math:`r`. In particular, the gradient of a vector field can only be calculated
        if the azimuthal component of the vector field vanishes. Similarly, the
        divergence of a tensor field can only be taken in special situations.
    """

    dim = 3  # dimension of the described space
    axes = ["r"]
    axes_symmetric = ["theta", "phi"]
    coordinate_constraints = [0, 1, 2]  # axes not described explicitly

    def point_to_cartesian(
        self, points: np.ndarray, *, full: bool = False
    ) -> np.ndarray:
        """convert coordinates of a point to Cartesian coordinates

        This function returns points along the z-coordinate, i.e, the x and y
        coordinates will be zero.

        Args:
            points (:class:`~numpy.ndarray`): The grid coordinates of the points
            full (bool): Flag indicating whether angular coordinates are specified

        Returns:
            :class:`~numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)
        if full:
            # angles are supplied
            if points.shape[-1] != self.dim:
                raise DimensionError(f"Shape {points.shape} cannot denote points")

            rsinθ = points[..., 0] * np.sin(points[..., 1])
            x = rsinθ * np.cos(points[..., 2])
            y = rsinθ * np.sin(points[..., 2])
            z = points[..., 0] * np.cos(points[..., 1])

        else:
            # angles are not supplied
            if points.shape[-1] != self.num_axes:
                raise DimensionError(f"Shape {points.shape} cannot denote points")
            z = points[..., 0]
            x = y = np.zeros_like(z)

        return np.stack((x, y, z), axis=-1)
