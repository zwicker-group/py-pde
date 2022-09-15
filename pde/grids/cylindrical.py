"""
Cylindrical grids with azimuthal symmetry

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generator, Sequence, Tuple, Union

import numpy as np

from ..tools.cache import cached_property
from .base import DimensionError, GridBase, _check_shape, discretize_interval
from .cartesian import CartesianGrid

if TYPE_CHECKING:
    from .boundaries.axes import Boundaries, BoundariesData  # @UnusedImport
    from .spherical import PolarSymGrid  # @UnusedImport


class CylindricalSymGrid(GridBase):  # lgtm [py/missing-equals]
    r""" 3-dimensional cylindrical grid assuming polar symmetry 

    The polar symmetry implies that states only depend on the radial and axial
    coordinates :math:`r` and :math:`z`, respectively. These are discretized uniformly as

    .. math::
        :nowrap:

        \begin{align*}
            r_i &= \left(i + \frac12\right) \Delta r
            &&\quad \text{for} \quad i = 0, \ldots, N_r - 1
            &&\quad \text{with} \quad \Delta r = \frac{R}{N_r}
        \\
            z_j &= z_\mathrm{min} + \left(j + \frac12\right) \Delta z
            &&\quad \text{for} \quad j = 0, \ldots, N_z - 1
            &&\quad \text{with}
                \quad \Delta z = \frac{z_\mathrm{max} - z_\mathrm{min}}{N_z}
        \end{align*}

    where :math:`R` is the radius of the cylindrical grid, :math:`z_\mathrm{min}` and
    :math:`z_\mathrm{max}` denote the respective lower and upper bounds of the axial
    direction, so that :math:`z_\mathrm{max} - z_\mathrm{min}` is the total height. The
    two axes are discretized by :math:`N_r` and :math:`N_z` support points, respectively.

    Warning:
        The order of components in the vector and tensor fields defined on this grid is
        different than in ordinary math. While it is common to use :math:`(r, \phi, z)`,
        we here use the order :math:`(r, z, \phi)`. It might thus be best to access
        components by name instead of index, e.g., use  :code:`field['z']` instead of
        :code:`field[1]`.
    """

    dim = 3  # dimension of the described space
    num_axes = 2  # number of independent axes
    axes = ["r", "z"]  # name of the actual axes
    axes_symmetric = ["phi"]
    coordinate_constraints = [0, 1]  # constraint Cartesian coordinates
    boundary_names = {  # name all the boundaries
        "inner": (0, False),
        "outer": (0, True),
        "bottom": (1, False),
        "top": (1, True),
    }

    def __init__(
        self,
        radius: float,
        bounds_z: Tuple[float, float],
        shape: Union[int, Sequence[int]],
        periodic_z: bool = False,
    ):
        """
        Args:
            radius (float):
                The radius of the cylinder
            bounds_z (tuple):
                The lower and upper bound of the z-axis
            shape (tuple):
                The number of support points in r and z direction, respectively. The same
                number is used for both if a single value is given.
            periodic_z (bool):
                Determines whether the z-axis has periodic boundary conditions.
        """
        super().__init__()
        shape_list = _check_shape(shape)
        if len(shape_list) == 1:
            self._shape: Tuple[int, int] = (shape_list[0], shape_list[0])
        elif len(shape_list) == 2:
            self._shape = tuple(shape_list)  # type: ignore
        else:
            raise DimensionError("`shape` must be two integers")
        if len(bounds_z) != 2:
            raise ValueError(
                "Lower and upper value of the axial coordinate must be specified"
            )
        self._periodic_z: bool = bool(periodic_z)  # might cast from np.bool_
        self.periodic = [False, self._periodic_z]

        # radial discretization
        dr = radius / self.shape[0]
        rs = (np.arange(self.shape[0]) + 0.5) * dr
        assert np.isclose(rs[-1] + dr / 2, radius)

        # axial discretization
        zs, dz = discretize_interval(*bounds_z, self.shape[1])
        assert np.isclose(zs[-1] + dz / 2, bounds_z[1])

        self._axes_coords = (rs, zs)
        self._axes_bounds = ((0.0, radius), tuple(bounds_z))  # type: ignore
        self._discretization = np.array((dr, dz))

    @property
    def state(self) -> Dict[str, Any]:
        """state: the state of the grid"""
        radius = self.axes_bounds[0][1]
        return {
            "radius": radius,
            "bounds_z": self.axes_bounds[1],
            "shape": self.shape,
            "periodic_z": self._periodic_z,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "CylindricalSymGrid":  # type: ignore
        """create a field from a stored `state`.

        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(
            radius=state_copy.pop("radius"),
            bounds_z=state_copy.pop("bounds_z"),
            shape=state_copy.pop("shape"),
            periodic_z=state_copy.pop("periodic_z"),
        )
        if state_copy:
            raise ValueError(f"State items {state_copy.keys()} were not used")
        return obj

    @classmethod
    def from_bounds(
        cls,
        bounds: Sequence[Tuple[float, float]],
        shape: Sequence[int],
        periodic: Sequence[bool],
    ) -> CylindricalSymGrid:
        """
        Args:
            bounds (tuple):
                Give the coordinate range for each axis. This should be a tuple of two
                number (lower and upper bound) for each axis. The length of `bounds`
                must be 2.
            shape (tuple):
                The number of support points for each axis. The length of `shape` needs
                to be 2.
            periodic (bool or list):
                Specifies which axes possess periodic boundary conditions. The first
                entry is ignored.

        Returns:
            CylindricalGrid representing the region chosen by bounds
        """
        radii, bounds_z = bounds
        if radii[0] != 0:
            raise NotImplementedError("Cylinders with hollow core are not implemented.")
        return cls(radii[1], bounds_z, shape, periodic_z=periodic[1])

    @property
    def radius(self) -> float:
        """float: radius of the cylinder"""
        return self.axes_bounds[0][1]

    @property
    def length(self) -> float:
        """float: length of the cylinder"""
        return self.axes_bounds[1][1] - self.axes_bounds[1][0]

    @property
    def volume(self) -> float:
        """float: total volume of the grid"""
        return float(np.pi * self.radius**2 * self.length)

    def get_random_point(
        self,
        *,
        boundary_distance: float = 0,
        avoid_center: bool = False,
        coords: str = "cartesian",
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """return a random point within the grid

        Note that these points will be uniformly distributed on the radial axis,
        which implies that they are not uniformly distributed in the volume.

        Args:
            boundary_distance (float): The minimal distance this point needs to
                have from all boundaries.
            avoid_center (bool): Determines whether the boundary distance
                should also be kept from the center, i.e., whether points close
                to the center are returned.
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
        r_min = boundary_distance if avoid_center else 0
        r_max = self.radius - boundary_distance
        z_min, z_max = self.axes_bounds[1]
        if boundary_distance != 0:
            z_min += boundary_distance
            z_max -= boundary_distance
            if r_max <= r_min or z_max <= z_min:
                raise RuntimeError("Random points would be too close to boundary")

        # create random point
        r = np.sqrt(rng.uniform(r_min**2, r_max**2))
        z = rng.uniform(z_min, z_max)
        if coords == "cartesian":
            φ = rng.uniform(0, 2 * np.pi)  # additional random angle
            return self.point_to_cartesian(np.array([r, z, φ]), full=True)

        elif coords == "cell":
            return self.transform(np.array([r, z]), "grid", "cell")

        elif coords == "grid":
            return np.array([r, z])

        else:
            raise ValueError(f"Unknown coordinate system `{coords}`")

    def get_line_data(self, data: np.ndarray, extract: str = "auto") -> Dict[str, Any]:
        """return a line cut for the cylindrical grid

        Args:
            data (:class:`~numpy.ndarray`):
                The values at the grid points
            extract (str):
                Determines which cut is done through the grid. Possible choices
                are (default is `cut_axial`):

                * `cut_z` or `cut_axial`: values along the axial coordinate for
                  :math:`r=0`.
                * `project_z` or `project_axial`: average values for each axial
                  position (radial average).
                * `project_r` or `project_radial`: average values for each
                  radial position (axial average)
        Returns:
            A dictionary with information about the line cut, which is
            convenient for plotting.
        """
        if extract == "auto":
            extract = "cut_axial"

        if extract == "cut_z" or extract == "cut_axial":
            # do a cut along the z axis for r=0
            axis = 1
            data_y: Union[np.ndarray, Tuple[np.ndarray]] = data[..., 0, :]
            label_y = "Cut along z"

        elif extract == "project_z" or extract == "project_axial":
            # project on the axial coordinate (average radially)
            axis = 1
            data_y = (data.mean(axis=-2),)
            label_y = "Projection onto z"

        elif extract == "project_r" or extract == "project_radial":
            # project on the radial coordinate (average axially)
            axis = 0
            data_y = (data.mean(axis=-1),)
            label_y = "Projection onto r"

        else:
            raise ValueError(f"Unknown extraction method {extract}")

        return {
            "data_x": self.axes_coords[axis],
            "data_y": data_y,
            "extent_x": self.axes_bounds[axis],
            "label_x": self.axes[axis],
            "label_y": label_y,
        }

    def get_image_data(self, data: np.ndarray) -> Dict[str, Any]:
        """return a 2d-image of the data

        Args:
            data (:class:`~numpy.ndarray`): The values at the grid points

        Returns:
            A dictionary with information about the image, which is  convenient
            for plotting.
        """
        bounds_r, bounds_z = self.axes_bounds
        return {
            "data": np.vstack((data[::-1, :], data)),
            "x": self.axes_coords[0],
            "y": self.axes_coords[1],
            "extent": (-bounds_r[1], bounds_r[1], bounds_z[0], bounds_z[1]),
            "label_x": self.axes[0],
            "label_y": self.axes[1],
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
        point = np.asanyarray(point, dtype=np.double)

        if with_self:
            yield point

        if not only_periodic or self._periodic_z:
            yield point - np.array([self.length, 0, 0])
            yield point + np.array([self.length, 0, 0])

    @cached_property()
    def cell_volume_data(self) -> Tuple[np.ndarray, float]:
        """:class:`~numpy.ndarray`: the volumes of all cells"""
        dr, dz = self.discretization
        rs = np.arange(self.shape[0] + 1) * dr
        areas = np.pi * rs**2
        r_vols = np.diff(areas).reshape(self.shape[0], 1)
        return (r_vols, dz)

    def point_to_cartesian(
        self, points: np.ndarray, *, full: bool = False
    ) -> np.ndarray:
        """convert coordinates of a point to Cartesian coordinates

        Args:
            points (:class:`~numpy.ndarray`): The grid coordinates of the points
            full (bool): Flag indicating whether angular coordinates are specified

        Returns:
            :class:`~numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)

        z = points[..., 1]
        if full:
            if points.shape[-1] != self.dim:
                raise DimensionError(f"Shape {points.shape} cannot denote full points")
            x = points[..., 0] * np.cos(points[..., 2])
            y = points[..., 0] * np.sin(points[..., 2])
        else:
            if points.shape[-1] != self.num_axes:
                raise DimensionError(f"Shape {points.shape} cannot denote grid points")
            x = points[..., 0]
            y = np.zeros_like(x)
        return np.stack((x, y, z), axis=-1)

    def point_from_cartesian(self, points: np.ndarray) -> np.ndarray:
        """convert points given in Cartesian coordinates to this grid

        This function returns points restricted to the x-z plane, i.e., the
        y-coordinate will be zero.

        Args:
            points (:class:`~numpy.ndarray`):
                Points given in Cartesian coordinates.

        Returns:
            :class:`~numpy.ndarray`: Points given in the coordinates of the grid
        """
        points = np.atleast_1d(points)
        assert points.shape[-1] == self.dim, f"Point must have {self.dim} coordinates"

        rs = np.hypot(points[..., 0], points[..., 1])
        zs = points[..., 2]
        return np.stack((rs, zs), axis=-1)

    def polar_coordinates_real(
        self, origin: np.ndarray, *, ret_angle: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """return spherical coordinates associated with the grid

        Args:
            origin (:class:`~numpy.ndarray`): Coordinates of the origin at which the polar
                coordinate system is anchored. Note that this must be of the
                form `[0, 0, z_val]`, where only `z_val` can be chosen freely.
            ret_angle (bool): Determines whether the azimuthal angle is returned
                alongside the distance. If `False` only the distance to the
                origin is  returned for each support point of the grid.
                If `True`, the distance and angles are returned.
        """
        origin = np.array(origin, dtype=np.double, ndmin=1)
        if len(origin) != self.dim:
            raise DimensionError("Dimensions are not compatible")

        if origin[0] != 0 or origin[1] != 0:
            raise RuntimeError("Origin must lie on symmetry axis for cylindrical grid")

        # calculate the difference vector between all cells and the origin
        diff = self.difference_vector_real(np.array([0, origin[2]]), self.cell_coords)
        dist: np.ndarray = np.linalg.norm(diff, axis=-1)  # get distance

        if ret_angle:
            return dist, np.arctan2(diff[:, :, 0], diff[:, :, 1])
        else:
            return dist

    def get_cartesian_grid(self, mode: str = "valid") -> CartesianGrid:
        """return a Cartesian grid for this Cylindrical one

        Args:
            mode (str):
                Determines how the grid is determined. Setting it to 'valid'
                only returns points that are fully resolved in the cylindrical
                grid, e.g., the cylinder is circumscribed. Conversely, 'full'
                returns all data, so the cylinder is inscribed.

        Returns:
            :class:`pde.grids.cartesian.CartesianGrid`: The requested grid
        """
        # Pick the grid instance
        if mode == "valid":
            bounds = self.radius / np.sqrt(self.dim)
        elif mode == "full":
            bounds = self.radius
        else:
            raise ValueError(f"Unsupported mode `{mode}`")

        # determine the Cartesian grid
        num = round(bounds / self.discretization[0])
        grid_bounds = [(-bounds, bounds), (-bounds, bounds), self.axes_bounds[1]]
        grid_shape = 2 * num, 2 * num, self.shape[1]
        return CartesianGrid(grid_bounds, grid_shape)

    def slice(self, indices: Sequence[int]) -> Union["CartesianGrid", "PolarSymGrid"]:
        """return a subgrid of only the specified axes

        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid

        Returns:
            :class:`~pde.grids.cartesian.CartesianGrid` or
            :class:`~pde.grids.spherical.PolarSymGrid`: The subgrid
        """
        if len(indices) != 1:
            raise ValueError(f"Can only get sub-grid for one axis.")

        if indices[0] == 0:
            # return a radial grid
            from .spherical import PolarSymGrid  # @Reimport

            return PolarSymGrid(self.radius, self.shape[0])

        elif indices[0] == 1:
            # return a Cartesian grid along the z-axis
            subgrid = CartesianGrid(
                bounds=[self.axes_bounds[1]],
                shape=self.shape[1],
                periodic=self.periodic[1],
            )
            subgrid.axes = [self.axes[1]]
            return subgrid

        else:
            raise ValueError(f"Cannot get sub-grid for index {indices[0]}")
