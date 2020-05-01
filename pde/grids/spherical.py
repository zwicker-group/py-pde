'''
Spherically-symmetric grids in 2 and 3 dimensions. These are grids that only
discretize the radial direction, assuming symmetry with respect to all angles.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
'''

from abc import ABCMeta
from typing import Tuple, Dict, Any, Union, Generator, TYPE_CHECKING

import numpy as np
from scipy import interpolate

from .base import GridBase, discretize_interval, _check_shape, DimensionError
from .cartesian import CartesianGrid
from ..tools.spherical import volume_from_radius
from ..tools.cache import cached_property
from ..tools.docstrings import fill_in_docstring



if TYPE_CHECKING:
    from .boundaries import Boundaries  # @UnusedImport



PI_4 = 4 * np.pi
PI_43 = 4 / 3 * np.pi



class SphericalGridBase(GridBase,  # lgtm [py/missing-equals]
                        metaclass=ABCMeta):
    r""" Base class for d-dimensional spherical grids with angular symmetry
    
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
    num_axes = 1        # the number of independent axes
    

    def __init__(self, radius: Union[float, Tuple[float, float]],
                 shape: Union[Tuple[int], int]):
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
            raise ValueError('`shape` must be a single number, not '
                             f'{shape_list}')
        self._shape: Tuple[int] = (int(shape_list[0]),)
        
        try:
            r_inner, r_outer = radius  # type: ignore
        except TypeError:
            r_inner, r_outer = 0, float(radius)  # type: ignore
            
        if r_inner < 0:
            raise ValueError('Inner radius must be positive')
        if r_inner >= r_outer:
            raise ValueError('Outer radius must be larger than inner radius')
        
        # radial discretization
        rs, dr = discretize_interval(r_inner, r_outer, self.shape[0])
        
        self._axes_coords = (rs,)
        self._axes_bounds = ((r_inner, r_outer),)
        self._discretization = np.array((dr,))
        
        
    @property
    def state(self) -> Dict[str, Any]:
        """ state: the state of the grid """
        return {'radius': self.radius,
                'shape': self.shape}
        
        
    @property
    def has_hole(self) -> bool:
        """ returns whether the inner radius is larger than zero """
        return self.axes_bounds[0][0] > 0
        
        
    @classmethod
    def from_state(cls, state) -> "SphericalGridBase":
        """ create a field from a stored `state`.
        
        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(radius=state_copy.pop('radius'),
                  shape=state_copy.pop('shape'))
        if state_copy:
            raise ValueError(f'State items {state_copy.keys()} were not used')
        return obj


    @property
    def radius(self) -> Union[float, Tuple[float, float]]:
        """ float: radius of the sphere """
        r_inner, r_outer = self.axes_bounds[0]
        if r_inner == 0:
            return r_outer
        else:
            return r_inner, r_outer
    

    @property
    def volume(self) -> float:
        """ float: total volume of the grid """
        r_inner, r_outer = self.axes_bounds[0]
        volume = volume_from_radius(r_outer, dim=self.dim)
        if r_inner > 0:
            volume -= volume_from_radius(r_inner, dim=self.dim)
        return volume


    @cached_property()
    def cell_volume_data(self) -> Tuple[np.ndarray]:
        """ tuple of :class:`numpy.ndarray`: the volumes of all cells """
        dr = self.discretization[0]
        rs = self.axes_coords[0]
        volumes_h = volume_from_radius(rs + 0.5 * dr, dim=self.dim)
        volumes_l = volume_from_radius(rs - 0.5 * dr, dim=self.dim)
        return ((volumes_h - volumes_l).reshape(self.shape[0]),)  # type: ignore
    

    def contains_point(self, point):
        """ check whether the point is contained in the grid
        
        Args:
            point (vector): Coordinates of the point
        """
        point = np.atleast_1d(point)
        if point.shape[-1] != self.dim:
            raise DimensionError('Dimension mismatch')
        r = np.linalg.norm(point, axis=-1)
        
        r_inner, r_outer = self.axes_bounds[0]
        return (r_inner <= r <= r_outer)

    
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
        r_inner, r_outer = self.axes_bounds[0]
        r_min = r_inner
        if avoid_center:
            r_min += boundary_distance
        r_mag = r_outer - boundary_distance - r_min

        if r_mag <= 0:
            raise RuntimeError('Random points would be too close to boundary')

        # create random point
        r = np.array([r_mag * np.random.random() + r_min])
        if cartesian:
            return self.point_to_cartesian(r)
        else:
            return r


    def get_line_data(self, data, extract: str = 'auto') -> Dict[str, Any]:
        """ return a line cut along the radial axis
        
        Args:
            data (:class:`numpy.ndarray`):
                The values at the grid points
            extract (str):
                Determines which cut is done through the grid. This parameter is
                mainly supplied for a consistent interface and has no effect for
                polar grids.
            
        Returns:
            A dictionary with information about the line cut, which is 
            convenient for plotting.
        """
        if extract not in {'auto', 'r', 'radial'}:
            raise ValueError(f'Unknown extraction method `{extract}`')
        
        return {'data_x': self.axes_coords[0],
                'data_y': data,
                'extent_x': self.axes_bounds[0],
                'label_x': self.axes[0]}


    def get_image_data(self, data, performance_goal: str = 'speed',
                       fill_value: float = 0,
                       masked: bool = True) -> Dict[str, Any]:
        """ return a 2d-image of the data
        
        Args:
            data (:class:`numpy.ndarray`):
                The values at the grid points
            performance_goal (str):
                Determines the method chosen for interpolation. Possible options
                are `speed` and `quality`.
            fill_value (float):
                The value assigned to invalid positions (those inside the hole
                or outside the region).
            masked (bool):
                Whether a :class:`numpy.ma.MaskedArray` is returned for the data
                instead of the normal :class:`numpy.ndarray`.
            
        Returns:
            A dictionary with information about the image, which is  convenient
            for plotting.
        """
        _, r_outer = self.axes_bounds[0]
        r_data = self.axes_coords[0]
        
        if self.has_hole:
            num = int(np.ceil(r_outer / self.discretization[0]))
            x_positive, _ = discretize_interval(0, r_outer, num)
        else:
            x_positive = r_data
            
        x = np.r_[-x_positive[::-1], x_positive]
        xs, ys = np.meshgrid(x, x, indexing='ij')
        r_img = np.hypot(xs, ys)
        
        if performance_goal == 'speed':
            # interpolate over the new coordinates using linear interpolation
            f = interpolate.interp1d(r_data, data, copy=False,
                                     bounds_error=False,
                                     fill_value=fill_value,
                                     assume_sorted=True)
            data_int = f(r_img.flat).reshape(r_img.shape)
            
        elif performance_goal == 'quality':
            # interpolate over the new coordinates using radial base function
            f = interpolate.Rbf(r_data, data, function='cubic')
            data_int = f(r_img)
            
        else:
            raise ValueError(f'Performance goal `{performance_goal}` undefined')
        
        if masked:
            mask = (r_img < r_data[0]) | (r_data[-1] < r_img) 
            data_int = np.ma.masked_array(data_int, mask=mask)
        
        return {'data': data_int,
                'x': x, 'y': x,
                'extent': (-r_outer, r_outer, -r_outer, r_outer),
                'label_x': 'x', 'label_y': 'y'}       


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


    def normalize_point(self, point, reduced_coords: bool = False):
        """ normalize coordinates, which is a no-op for spherical coordinates.
        
        Args:
            point (:class:`numpy.ndarray`): Coordinates of a single point
            reduced_coords (bool): Flag determining whether only the coordinates
                corresponding to axes in this grid are given
                 
        Returns:
            :class:`numpy.ndarray`: The input array
        """
        point = np.asarray(point, dtype=np.double)
        size = self.num_axes if reduced_coords else self.dim
        if point.size == 0:
            return np.zeros((0, size))
        if point.shape[-1] != size:
            raise DimensionError('Dimension mismatch: Array of shape '
                                 f'{point.shape} does not describe points of '
                                 f'dimension {size}.')
        return point

    
    def point_from_cartesian(self, points):
        """ convert points given in Cartesian coordinates to this grid
        
        Args:
            points (:class:`numpy.ndarray`):
                Points given in Cartesian coordinates.
                
        Returns:
            :class:`numpy.ndarray`: Points given in the coordinates of the grid
        """
        points = self.normalize_point(points)
        return np.linalg.norm(points, axis=-1, keepdims=True)


    def cell_to_point(self, cells, cartesian: bool = True):
        """ convert cell coordinates to real coordinates
        
        This function returns points restricted to the x-axis, i.e., the
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
        # convert from cells indices to grid coordinates
        r_inner, _ = self.axes_bounds[0] 
        points = r_inner + (cells + 0.5) * self.discretization[0]
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
        # convert from grid coordinates to cells indices
        r_inner, _ = self.axes_bounds[0]
        r = self.point_from_cartesian(points)
        cells = (r - r_inner) / self.discretization[0]
        return cells.astype(np.int)

        
    def difference_vector_real(self, p1, p2):
        """ return the vector pointing from p1 to p2.
        
        In case of periodic boundary conditions, the shortest vector is returned
        
        Args:
            p1 (:class:`numpy.ndarray`): First point(s)
            p2 (:class:`numpy.ndarray`): Second point(s)
            
        Returns:
            :class:`numpy.ndarray`: The difference vectors between the points
                with periodic boundary conditions applied.
        """
        return np.atleast_1d(p2) - np.atleast_1d(p1)

    
    def polar_coordinates_real(self, origin=[0, 0, 0], ret_angle: bool = False):
        """ return spherical coordinates associated with the grid
        
        Args:
            origin (vector): Coordinates of the origin at which the polar
                coordinate system is anchored. Note that this must be of the 
                form `[0, 0, z_val]`, where only `z_val` can be chosen freely. 
            ret_angle (bool): Determines whether angles are returned alongside
                the distance. If `False` only the distance to the origin is 
                returned for each support point of the grid.
                If `True`, the distance and angles are returned. Note that in 
                the case of spherical grids, this angle is zero by convention.
        """
        origin = np.array(origin, dtype=np.double, ndmin=1)
        if not np.array_equal(origin, np.zeros(self.dim)):
            raise RuntimeError(f'Origin must be {str([0]*self.dim)}')
             
        # the distance to the origin is exactly the radial coordinate
        rs = self.axes_coords[0]
        if ret_angle:
            return rs, np.zeros_like(rs)
        else:
            return rs


    @fill_in_docstring
    def get_boundary_conditions(self, bc='natural', rank: int = 0) \
            -> "Boundaries":
        """ constructs boundary conditions from a flexible data format.
        
        Note that the inner boundary condition for grids without holes need not
        be specified.
        
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
        from .boundaries.local import BCBase, NeumannBC
        from .boundaries.axis import BoundaryPair
        from .boundaries import Boundaries  # @Reimport

        if isinstance(bc, Boundaries):
            return bc

        if self.has_hole:  # grid has holes => specify two boundary conditions
            if bc == 'natural':
                low = NeumannBC(self, 0, upper=False, rank=rank)
                high = NeumannBC(self, 0, upper=True, rank=rank)
                b_pair = BoundaryPair(low, high)
            else:
                b_pair = BoundaryPair.from_data(self, 0, bc, rank=rank)
            
        else:  # grid has no hole => need only one boundary condition
            b_inner = NeumannBC(self, 0, upper=False, rank=rank)
            if bc == 'natural':
                upper = NeumannBC(self, 0, upper=True, rank=rank)
                b_pair = BoundaryPair(b_inner, upper)
            else:
                try:
                    b_outer = BCBase.from_data(self, 0, upper=True, data=bc,
                                               rank=rank)
                except ValueError:
                    # this can happen when two boundary conditions have been
                    # supplied
                    b_pair = BoundaryPair.from_data(self, 0, bc, rank=rank)
                    if b_inner != b_pair.low:
                        raise ValueError(
                            f'Unsupported boundary format: `{bc}`. Note that '
                            'spherical symmetry implies vanishing derivatives '
                            'at the origin at r=0 for all fields. This '
                            'boundary condition need not be specified.')
                else:
                    b_pair = BoundaryPair(b_inner, b_outer)

        return Boundaries([b_pair])       

    
    def get_cartesian_grid(self, mode: str = 'valid', num: int = None) \
            -> CartesianGrid:
        """ return a Cartesian grid for this Cylindrical one 
        
        Args:
            mode (str):
                Determines how the grid is determined. Setting it to 'valid'
                only returns points that are fully resolved in the spherical
                grid, e.g., the sphere is circumscribed. Conversely, 'full'
                returns all data, so the sphere is inscribed. 
            num (int):
                Number of support points along each axis of the returned grid.
                
        Returns:
            :class:`pde.grids.cartesian.CartesianGrid`: The requested grid
        """
        # Pick the grid instance
        if mode == 'valid':
            if self.has_hole:
                self._logger.warn('The sphere has holes, so not all points are '
                                  'valid.')
            bounds = self.radius / np.sqrt(self.dim)
        elif mode == 'full':
            bounds = self.radius
        else:
            raise ValueError(f'Unsupported mode `{mode}`')
                
        # determine the grid points
        if num is None:
            num = 2 * round(bounds / self.discretization[0])
        grid_bounds = [(-bounds, bounds)] * self.dim
        return CartesianGrid(grid_bounds, num)
        


class PolarGrid(SphericalGridBase):
    r""" 2-dimensional polar grid assuming angular symmetry
    
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
    axes = ['r']
    axes_symmetric = ['phi']
    coordinate_constraints = [0, 1]  # axes not described explicitly

        
    def point_to_cartesian(self, points):
        """ convert coordinates of a point to Cartesian coordinates
        
        This function returns points along the y-coordinate, i.e, the x
        coordinates will be zero.
                
        Returns:
            :class:`numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != 1:
            raise DimensionError(f'Dimension mismatch: Points {points} invalid')
        
        y = points[..., 0]
        x = np.zeros_like(y)
        return np.stack((x, y), axis=-1)


    def plot(self, title: str = None, show: bool = False, **kwargs):
        r""" visualize the polar grid
        
        Args:
            title (str):
                Determines the title of the figure
            show (bool):
                Determines whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Extra arguments are passed on the to the matplotlib
                plotting routines, e.g., to set the color of the lines
        """
        import matplotlib.pyplot as plt
        from ..visualization.plotting import finalize_plot
        
        kwargs.setdefault('color', 'k')
        rb, = self.axes_bounds
        rmax = rb[1]
        ax = plt.gca()
        for r in np.linspace(*rb, self.shape[0] + 1):
            if r == 0:
                plt.plot(0, 0, '.', **kwargs)
            else:
                c = plt.Circle((0, 0), r, fill=False, **kwargs)
                ax.add_artist(c)
        plt.xlim(-rmax, rmax)
        plt.xlabel('x')
        plt.ylim(-rmax, rmax)
        plt.ylabel('y')
        ax.set_aspect(1)
        
        finalize_plot(title=title, show=show)
            
            
    
class SphericalGrid(SphericalGridBase):
    r""" 3-dimensional spherical grid assuming spherical symmetry
    
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
    """
    
    dim = 3  # dimension of the described space
    axes = ['r']
    axes_symmetric = ['theta', 'phi']
    coordinate_constraints = [0, 1, 2]  # axes not described explicitly
      
      
    def point_to_cartesian(self, points):
        """ convert coordinates of a point to Cartesian coordinates
        
        This function returns points along the z-coordinate, i.e, the x and y
        coordinates will be zero.
        
        Args:
            points (:class:`numpy.ndarray`):
                Points given in the coordinates of the grid
                
        Returns:
            :class:`numpy.ndarray`: The Cartesian coordinates of the point
        """
        points = np.atleast_1d(points)
        if points.shape[-1] != 1:
            raise DimensionError(f'Dimension mismatch: Points {points} invalid')
        z = points[..., 0]
        x = np.zeros_like(z)
        return np.stack((x, x, z), axis=-1)    
