'''
Cylindrical grids with azimuthal symmetry

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
'''

from typing import Tuple, Sequence, Dict, Union, Any, Generator, TYPE_CHECKING

import numpy as np

from .base import GridBase, discretize_interval, _check_shape, DimensionError
from .cartesian import CartesianGrid
from ..tools.cache import cached_property
from ..tools.docstrings import fill_in_docstring


if TYPE_CHECKING:
    from .boundaries import Boundaries  # @UnusedImport
    from .spherical import PolarGrid  # @UnusedImport


    
class CylindricalGrid(GridBase):  # lgtm [py/missing-equals]
    r""" 3-dimensional cylindrical grid assuming polar symmetry 
    
    The polar symmetry implies that states only depend on the radial and axial
    coordinates :math:`r` and :math:`z`, respectively. These are discretized
    uniformly as
    
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
    
    where :math:`R` is the radius of the cylindrical grid,
    :math:`z_\mathrm{min}` and :math:`z_\mathrm{max}` denote the respective
    lower and upper bounds of the axial direction, so that
    :math:`z_\mathrm{max} - z_\mathrm{min}` is the total height.
    The two axes are discretized by :math:`N_r` and :math:`N_z` support points,
    respectively.
    """
    
    dim = 3            # dimension of the described space
    num_axes = 2       # number of independent axes
    axes = ['r', 'z']  # name of the actual axes
    axes_symmetric = ['phi']
    coordinate_constraints = [0, 1]  # constraint Cartesian coordinates 
    
    
    def __init__(self, radius: float,
                 bounds_z: Tuple[float, float],
                 shape: Tuple[int, int],
                 periodic_z: bool = False):
        """ 
        Args:
            radius (float): The radius of the cylinder
            bounds_z (tuple): The lower and upper bound of the z-axis
            shape (tuple): The number of support points in r and z direction,
                respectively.
            periodic_z (bool): Determines whether the z-axis has periodic
                boundary conditions.
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
            raise ValueError('Lower and upper value of the axial coordinate '
                             'must be specified')
        self._periodic_z: bool = bool(periodic_z)  # might cast from np.bool_
        self.periodic = [False, self._periodic_z]

        # radial discretization
        dr = radius / self.shape[0]
        rs = (np.arange(self.shape[0]) + 0.5) * dr
        assert np.isclose(rs[-1] + dr/2, radius)
        
        # axial discretization
        zs, dz = discretize_interval(*bounds_z, self.shape[1])
        assert np.isclose(zs[-1] + dz/2, bounds_z[1])
        
        self._axes_coords = (rs, zs)
        self._axes_bounds = ((0., radius), tuple(bounds_z))  # type: ignore 
        self._discretization = np.array((dr, dz))
        
        
    @property
    def state(self) -> Dict[str, Any]:
        """ state: the state of the grid """
        radius = self.axes_bounds[0][1]
        return {'radius': radius,
                'bounds_z': self.axes_bounds[1],
                'shape': self.shape,
                'periodic_z': self._periodic_z}
        
        
    @classmethod
    def from_state(cls,  # type: ignore
                   state: Dict[str, Any]) -> "CylindricalGrid":
        """ create a field from a stored `state`.
        
        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(radius=state_copy.pop('radius'),
                  bounds_z=state_copy.pop('bounds_z'),
                  shape=state_copy.pop('shape'),
                  periodic_z=state_copy.pop('periodic_z'))
        if state_copy:
            raise ValueError(f'State items {state_copy.keys()} were not used')
        return obj
        
        
    @property
    def radius(self) -> float:
        """ float: radius of the cylinder """
        return self.axes_bounds[0][1]        
        

    @property
    def length(self) -> float:
        """ float: length of the cylinder """
        return self.axes_bounds[1][1] - self.axes_bounds[1][0]        
        

    @property
    def volume(self) -> float:
        """ float: total volume of the grid """
        return float(np.pi * self.radius**2 * self.length) 


    def get_random_point(self, boundary_distance: float = 0,
                         cartesian: bool = True, avoid_center: bool = False):
        """ return a random point within the grid
        
        Note that these points will be uniformly distributed on the radial axis,
        which implies that they are not uniformly distributed in the volume.
        
        Args:
            boundary_distance (float): The minimal distance this point needs to
                have from all boundaries.
            cartesian (bool): Determines whether the point is returned in
                Cartesian coordinates or grid coordinates.
            avoid_center (bool): Determines whether the boundary distance
                should also be kept from the center, i.e., whether points close
                to the center are returned.
                
        Returns:
            :class:`numpy.ndarray`: The coordinates of the point        
        """
        # handle the boundary distance
        r_min = boundary_distance if avoid_center else 0
        r_mag = self.radius - boundary_distance - r_min
        z_min, z_max = self.axes_bounds[1]

        if boundary_distance != 0:
            z_min += boundary_distance
            z_max -= boundary_distance
            if r_mag <= 0 or z_max <= z_min:
                raise RuntimeError('Random points would be too close to '
                                   'boundary')

        # create random point
        r = r_mag * np.random.random() + r_min
        z = z_min + (z_max - z_min) * np.random.random()
        point = np.array([r, z])
        if cartesian:
            return self.point_to_cartesian(point)
        else:
            return point
        
    
    def get_line_data(self, data, extract: str = 'auto') -> Dict[str, Any]:
        """ return a line cut along the cylindrical symmetry axis
        
        Args:
            data (:class:`numpy.ndarray`):
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
        if extract == 'auto':
            extract = 'cut_axial'
        
        if extract == 'cut_z' or extract == 'cut_axial':
            # do a cut along the z axis for r=0
            axis = 1
            data_y = data[..., 0, :]
            label_y = 'Cut along z'
            
        elif extract == 'project_z' or extract == 'project_axial':
            # project on the axial coordinate (average radially)
            axis = 1
            data_y = data.mean(axis=-2),
            label_y = 'Projection onto z'
            
        elif extract == 'project_r' or extract == 'project_radial':
            # project on the radial coordinate (average axially)
            axis = 0
            data_y = data.mean(axis=-1),
            label_y = 'Projection onto r'
            
        else:
            raise ValueError(f'Unknown extraction method {extract}')
        
        return {'data_x': self.axes_coords[axis],
                'data_y': data_y,
                'extent_x': self.axes_bounds[axis],
                'label_x': self.axes[axis],
                'label_y': label_y}
    


    def get_image_data(self, data) -> Dict[str, Any]:
        """ return a 2d-image of the data
        
        Args:
            data (:class:`numpy.ndarray`): The values at the grid points
            
        Returns:
            A dictionary with information about the image, which is  convenient
            for plotting.
        """
        bounds_r, bounds_z = self.axes_bounds
        return {'data': np.vstack((data[::-1, :], data)).T,
                'x': self.axes_coords[0],
                'y': self.axes_coords[1],
                'extent': (-bounds_r[1], bounds_r[1], bounds_z[0], bounds_z[1]),
                'label_x': self.axes[0],
                'label_y': self.axes[1]}        
      
        
    def contains_point(self, point) -> bool:
        """ check whether the point is contained in the grid
        
        Args:
            point (vector): Coordinates of the point
        """
        assert len(point) == 3
        r = np.hypot(point[0], point[1])
        bounds_z = self.axes_bounds[1]
        return (r <= self.radius and  # type: ignore
                bounds_z[0] <= point[2] <= bounds_z[1])  


    def iter_mirror_points(self, point, with_self: bool = False,
                           only_periodic: bool = True) -> Generator:
        """ generates all mirror points corresponding to `point`
        
        Args:
            point (:class:`numpy.ndarray`): the point within the grid
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
        """ :class:`numpy.ndarray`: the volumes of all cells """
        dr, dz = self.discretization
        rs = np.arange(self.shape[0] + 1) * dr
        areas = np.pi * rs**2
        r_vols = np.diff(areas).reshape(self.shape[0], 1)
        return (r_vols, dz)
    
            
    def normalize_point(self, point, reduced_coords: bool = False):
        """ normalize coordinates by applying periodic boundary conditions
        
        Args:
            point (:class:`numpy.ndarray`): Coordinates of a single point
            reduced_coords (bool): Flag determining whether only the coordinates
                corresponding to axes in this grid are given
                 
        Returns:
            :class:`numpy.ndarray`: The respective coordinates with periodic
            boundary conditions applied.
        """
        point = np.asarray(point, dtype=np.double)
        size = self.num_axes if reduced_coords else self.dim
        if point.size == 0:
            return np.zeros((0, size))
        if point.shape[-1] != size:
            raise DimensionError('Dimension mismatch: Array of shape '
                                 f'{point.shape} does not describe points of '
                                 f'dimension {size}.')
        
        if self._periodic_z:
            z_min = self.axes_bounds[1][0]
            point[..., -1] = (point[..., -1] - z_min) % self.length + z_min
            return point
        else:
            return point
        
        
    def point_to_cartesian(self, points):
        """ convert coordinates of a point to Cartesian coordinates
        
        Args:
            points (:class:`numpy.ndarray`):
                Points given in the coordinates of the grid
                
        Returns:
            :class:`numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != 2:
            raise DimensionError(f'Dimension mismatch: Points {points} invalid')
        
        x = points[..., 0]
        y = np.zeros_like(x)
        z = points[..., 1] 
        return np.stack((x, y, z), axis=-1)
    
    
    def point_from_cartesian(self, points):
        """ convert points given in Cartesian coordinates to this grid
        
        This function returns points restricted to the x-z plane, i.e., the
        y-coordinate will be zero.
        
        Args:
            points (:class:`numpy.ndarray`):
                Points given in Cartesian coordinates.
                
        Returns:
            :class:`numpy.ndarray`: Points given in the coordinates of the grid
        """
        points = self.normalize_point(points)
        rs = np.hypot(points[..., 0], points[..., 1])
        zs = points[..., 2]
        return np.stack((rs, zs), axis=-1)


    def cell_to_point(self, cells, cartesian: bool = True):
        """ convert cell coordinates to real coordinates
        
        This function returns points restricted to the x-z plane, i.e., the
        y-coordinate will be zero.
        
        Args:
            cells (:class:`numpy.ndarray`):
                Indices of the cells whose center coordinates are requested.
                This can be float values to indicate positions relative to the
                cell center.
            cartesian (bool):
                Determines whether the point is returned in Cartesian
                coordinates or grid coordinates.
                
        Returns:
            :class:`numpy.ndarray`: The center points of the respective cells
        """
        cells = np.atleast_1d(cells)
        if cells.size == 0:
            return np.zeros((0, self.dim))
        if cells.shape[-1] != 2:
            raise DimensionError(f'Dimension mismatch: Cell {cells} invalid')

        # convert from cells indices to grid coordinates
        points = (cells + 0.5) * self.discretization
        points[..., 1] += self.axes_bounds[1][0]
        if cartesian:
            return self.point_to_cartesian(points)
        else:
            return points


    def point_to_cell(self, points):
        """ Determine cell(s) corresponding to given point(s)

        This function respects periodic boundary conditions, but it does not
        throw an error when coordinates lie outside the bcs (for
        non-periodic axes).
        
        Args:
            points (:class:`numpy.ndarray`): Real coordinates
                
        Returns:
            :class:`numpy.ndarray`: The indices of the respective cells
        """
        points = self.point_from_cartesian(points)
        # convert from grid coordinates to cells indices
        points[..., 1] -= self.axes_bounds[1][0]
        points /= self.discretization
        return points.astype(np.int)

        
    def difference_vector_real(self, p1, p2):
        """ return the vector pointing from p1 to p2.
        
        In case of periodic boundary conditions, the shortest vector is returned
        
        Args:
            p1 (:class:`numpy.ndarray`): First point(s)
            p2 (:class:`numpy.ndarray`): Second point(s)
            
        Returns:
            :class:`numpy.ndarray`: The difference vectors between the points
            with periodic  boundary conditions applied.
        """
        diff = np.atleast_1d(p2) - np.atleast_1d(p1)
        if self._periodic_z:
            size = self.length
            diff[..., 1] = (diff[..., 1] + size/2) % size - size/2
        return diff

    
    def polar_coordinates_real(self, origin, ret_angle: bool = False):
        """ return spherical coordinates associated with the grid
        
        Args:
            origin (vector): Coordinates of the origin at which the polar
                coordinate system is anchored. Note that this must be of the 
                form `[0, 0, z_val]`, where only `z_val` can be chosen freely. 
            ret_angle (bool): Determines whether the azimuthal angle is returned
                alongside the distance. If `False` only the distance to the
                origin is  returned for each support point of the grid.
                If `True`, the distance and angles are returned.
        """
        origin = np.array(origin, dtype=np.double, ndmin=1)
        if len(origin) != self.dim:
            raise DimensionError('Dimensions are not compatible')
            
        if origin[0] != 0 or origin[1] != 0:
            raise RuntimeError('Origin must lie on symmetry axis for '
                               'cylindrical grid')
             
        # calculate the difference vector between all cells and the origin
        diff = self.difference_vector_real([0, origin[2]], self.cell_coords)
        dist = np.linalg.norm(diff, axis=-1)  # get distance
        
        if ret_angle:
            return dist, np.arctan2(diff[:, :, 0], diff[:, :, 1])
        else:
            return dist


    @fill_in_docstring
    def get_boundary_conditions(self, bc='natural', rank: int = 0) \
            -> "Boundaries":
        """ constructs boundary conditions from a flexible data format
        
        Args:
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}  
            rank (int):
                The tensorial rank of the value associated with the boundary
                conditions.

        Raises:
            ValueError: If the data given in `bc` cannot be read
            PeriodicityError: If the boundaries are not compatible with the
                periodic axes of the grid. 
        """
        from .boundaries import Boundaries  # @Reimport
        return Boundaries.from_data(self, bc, rank=rank)
    
    
    
    def get_cartesian_grid(self, mode: str = 'valid') -> CartesianGrid:
        """ return a Cartesian grid for this Cylindrical one 
        
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
        if mode == 'valid':
            bounds = self.radius / np.sqrt(self.dim)
        elif mode == 'full':
            bounds = self.radius
        else:
            raise ValueError(f'Unsupported mode `{mode}`')
                
        # determine the Cartesian grid
        num = round(bounds / self.discretization[0])
        grid_bounds = [(-bounds, bounds), (-bounds, bounds),
                       self.axes_bounds[1]]
        grid_shape = 2 * num, 2 * num, self.shape[1]
        return CartesianGrid(grid_bounds, grid_shape)
    

    def get_subgrid(self, indices: Sequence[int]) \
            -> Union["CartesianGrid", "PolarGrid"]:
        """ return a subgrid of only the specified axes
        
        
        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid
                
        Returns:
            :class:`~pde.grids.cartesian.CartesianGrid` or 
            :class:`~pde.grids.spherical.PolarGrid`: The subgrid
        """
        if len(indices) != 1:
            raise ValueError(f'Can only get sub-grid for one axis.')
        
        if indices[0] == 0:
            # return a radial grid
            assert self.axes[0] == 'r'
            from .spherical import PolarGrid  # @Reimport
            return PolarGrid(self.radius, self.shape[0])
            
        elif indices[0] == 1:
            # return a Cartesian grid along the z-axis
            assert self.axes[1] == 'z'
            subgrid = CartesianGrid(bounds=[self.axes_bounds[1]],
                                    shape=self.shape[1],
                                    periodic=self.periodic[1])
            subgrid.axes = ['z']
            return subgrid
            
        else:
            raise ValueError(f'Cannot get sub-grid for index {indices[0]}')
