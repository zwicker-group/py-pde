'''
Cartesian grids of arbitrary dimension.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
'''

import itertools
from abc import ABCMeta
from typing import List, Sequence, Dict, Any, Union, Generator, TYPE_CHECKING

import numpy as np


from .base import GridBase, _check_shape, DimensionError
from ..tools.cuboid import Cuboid
from ..tools.docstrings import fill_in_docstring


if TYPE_CHECKING:
    from .boundaries import Boundaries  # @UnusedImport
    


class CartesianGridBase(GridBase,  # lgtm [py/missing-equals]
                        metaclass=ABCMeta):
    """Base class for :class:`UnitGrid` and :class:`CartesianGrid`"""
    
    cuboid: Cuboid
    
    
    def __init__(self, shape, periodic: Union[List[bool], bool] = False):
        """ 
        Args:
            shape (tuple): The number of support points for each axis. The
                dimension of the grid is given by `len(shape)`.
            periodic (bool or list): Specifies which axes possess periodic
                boundary conditions. This is either a list of booleans defining
                periodicity for each individual axis or a single boolean value
                specifying the same periodicity for all axes.
        """
        super().__init__()
        self._shape = _check_shape(shape)
        self.dim = len(self.shape)
        self.num_axes = self.dim
        
        if isinstance(periodic, (bool, np.bool_)):
            self.periodic = [bool(periodic)] * self.dim
        elif len(periodic) != self.dim:
            raise DimensionError('Number of axes with specified periodicity '
                                 'does not match grid dimension '
                                 f'({len(periodic)} != {self.dim})')
        else:
            self.periodic = periodic
        
        if self.dim <= 3:
            self.axes = list('xyz'[:self.dim])
        else:
            self.axes = [chr(97 + i) for i in range(self.dim)]
        

    @property
    def cell_volume_data(self):
        """ size associated with each cell """
        return tuple(self.discretization)
        

    def contains_point(self, point):
        """ check whether the point is contained in the grid
        
        Args:
            point (vector): Coordinates of the point
        """
        if len(point) != self.dim:
            raise DimensionError('Incompatible dimensions')
        return self.cuboid.contains_point(point)


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
        
        # find all offsets of the individual axes
        offsets = []
        for i in range(self.dim):
            if only_periodic and not self.periodic[i]:
                offsets.append([0])
            else:
                s = self.cuboid.size[i]
                offsets.append([-s, 0, s])
               
        # produce the respective mirrored points 
        for offset in itertools.product(*offsets):
            if with_self or np.linalg.norm(offset) != 0:
                yield point + offset
    
    
    def get_random_point(self, boundary_distance: float = 0,
                         cartesian: bool = True):
        """ return a random point within the grid
        
        Args:
            boundary_distance (float): The minimal distance this point needs to
                have from all boundaries.
            cartesian (bool): Determines whether the point is returned in
                Cartesian coordinates or grid coordinates. This does not have
                any effect for Cartesian coordinate systems, but the argument is
                retained to have a unified interface for all grids.
                
        Returns:
            :class:`numpy.ndarray`: The coordinates of the point        
        """
        # handle the boundary distance
        cuboid = self.cuboid
        if boundary_distance != 0:
            if any(cuboid.size <= 2 * boundary_distance):
                raise RuntimeError('Random points would be too close to '
                                   'boundary')
            cuboid = cuboid.buffer(-boundary_distance)
            
        # create random point
        return cuboid.pos + np.random.random(self.dim) * cuboid.size


    def get_line_data(self, data: np.ndarray, extract: str = 'auto') \
            -> Dict[str, Any]:
        """ return a line cut through the given data
        
        Args:
            data (:class:`numpy.ndarray`):
                The values at the grid points
            extract (str):
                Determines which cut is done through the grid. Possible choices
                are (default is `cut_0`):
                
                * `cut_#`: return values along the axis specified by # and use
                  the mid point along all other axes.
                * `project_#`: average values for all axes, except axis #.

                Here, # can either be a zero-based index (from 0 to dim-1) or
                a letter denoting the axis.
            
        Returns:
            A dictionary with information about the line cut, which is 
            convenient for plotting.
        """
        if data.shape[-self.dim:] != self.shape:
            raise ValueError(f'Shape {data.shape} of the data array is not '
                             f'compatible with grid shape {self.shape}')
            
        def _get_axis(axis):
            """ determine the axis from a given specifier """
            try:
                axis = int(axis)
            except ValueError:
                try:
                    axis = self.axes.index(axis)
                except ValueError:
                    raise ValueError(f'Axis `{axis}` not defined')
            return axis
            
        if extract == 'auto':
            extract = 'cut_0'  # use a cut along first axis
            
        if extract.startswith('cut_'):
            # consider a cut along a given axis
            axis = _get_axis(extract[4:])
            data_y = data
            rank = data.ndim - self.dim  # rank of data 
            for ax in reversed(range(self.dim)):
                if ax != axis:
                    mid_point = self.shape[ax] // 2
                    data_y = np.take(data_y, mid_point, axis=ax + rank)
            label_y = f'Cut along {self.axes[axis]}'
                
        elif extract.startswith('project_'):
            # consider a projection along a given axis
            axis = _get_axis(extract[8:])
            avg_axes = [ax - self.dim
                        for ax in range(self.dim)
                        if ax != axis]
            data_y = data.mean(axis=tuple(avg_axes))
            label_y = f'Projection onto {self.axes[axis]}'
                
        else:
            raise ValueError(f'Unknown extraction method `{extract}`')

        if self.dim == 1:
            label_y = ''

        # return the data with the respective labels                
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
        if data.shape[-self.dim:] != self.shape:
            raise ValueError(f'Shape {data.shape} of the data array is not '
                             f'compatible with grid shape {self.shape}')

        if self.dim == 2:
            image_data = data.T
        elif self.dim == 3: 
            image_data = data[:, :, self.shape[-1] // 2].T
        else:
            raise NotImplementedError('Creating images is only implemented for '
                                      '2d and 3d grids')
            
        extent: List[float] = []
        for c in self.axes_bounds[:2]:
            extent.extend(c)
                
        return {'data': image_data,
                'x': self.axes_coords[-2],
                'y': self.axes_coords[-1],
                'extent': extent,
                'label_x': 'x', 'label_y': 'y'}    


    def point_to_cartesian(self, points):
        """ convert coordinates of a point to Cartesian coordinates
        
        Args:
            points (:class:`numpy.ndarray`):
                Points given in the coordinates of \the grid
                
        Returns:
            :class:`numpy.ndarray`: The Cartesian coordinates of the point
        """
        return points
    
    
    def point_from_cartesian(self, coords):
        """ convert points given in Cartesian coordinates to this grid
        
        Args:
            points (:class:`numpy.ndarray`): Points in Cartesian coordinates.
                
        Returns:
            :class:`numpy.ndarray`: Points given in the coordinates of the grid
        """
        return coords

    
    def polar_coordinates_real(self, origin, ret_angle: bool = False):
        """ return polar coordinates associated with the grid
        
        Args:
            origin (vector): Coordinates of the origin at which the polar
                coordinate system is anchored.
            ret_angle (bool): Determines whether angles are returned alongside
                the distance. If `False` only the distance to the origin is 
                returned for each support point of the grid.
                If `True`, the distance and angles are returned. For a 1d system
                system, the angle is defined as the sign of the difference
                between the point and the origin, so that angles can either be
                1 or -1. For 2d systems and 3d systems, polar coordinates and
                spherical coordinates are used, respectively. 
        """
        origin = np.array(origin, dtype=np.double, ndmin=1)
        if len(origin) != self.dim:
            raise DimensionError('Dimensions are not compatible')
        
        # calculate the difference vector between all cells and the origin
        diff = self.difference_vector_real(origin, self.cell_coords)
        dist = np.linalg.norm(diff, axis=-1)  # get distance
        
        # determine distance and optionally angles for these vectors
        if ret_angle:
            if self.dim == 1:
                return dist, np.sign(diff)[..., 0]
            
            elif self.dim == 2:
                return dist, np.arctan2(diff[:, :, 0], diff[:, :, 1])
            
            elif self.dim == 3:
                theta = np.arccos(diff[..., 2] / dist)
                phi = np.arctan2(diff[..., 0], diff[..., 1])
                return dist, theta, phi 
            
            else:
                raise NotImplementedError('Cannot calculate angles for '
                                          f'dimension {self.dim}')
        else:
            return dist
        
        
    def from_polar_coordinates(self, distance, angle, origin=None):
        """ convert polar coordinates to Cartesian coordinates
        
        This function is currently only implemented for 1d and 2d systems.
        
        Args:
            distance: The radial distance
            angle: The angle with respect to the origin
            origin (optional): Sets the origin of the coordinate system. If
                omitted. The zero point is assumed as the origin.
                
        Returns:
            The Cartesian coordinates corresponding to the given polar
            coordinates. 
        """
        distance = np.asarray(distance)
        angle = np.asarray(angle)
        if origin is None:
            origin = np.zeros(self.dim)
        else:
            origin = np.atleast_1d(origin)
        
        if self.dim == 1:
            diff = distance * angle
            coords = origin + diff[..., None]
        
        elif self.dim == 2:
            unit_vector = np.moveaxis([np.sin(angle), np.cos(angle)], 0, -1)
            diff = distance[..., None] * unit_vector
            coords = origin + diff
        
        else:
            raise NotImplementedError('Cannot calculate coordinates for '
                                      f'dimension {self.dim}')
            
        return self.normalize_point(coords)


    def plot(self, title: str = None, show: bool = False, **kwargs):
        r""" visualize the grid
        
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
        
        if self.dim not in {1, 2}:
            raise NotImplementedError('Plotting is not implemented for grids '
                                      f'of dimension {self.dim}')
            
        kwargs.setdefault('color', 'k')
        xb = self.axes_bounds[0]
        for x in np.linspace(*xb, self.shape[0] + 1):
            plt.axvline(x, **kwargs)
        plt.xlim(*xb)
        plt.xlabel(self.axes[0])
        
        if self.dim == 2:
            yb = self.axes_bounds[1]
            for y in np.linspace(*yb, self.shape[1] + 1):
                plt.axhline(y, **kwargs)
            plt.ylim(*yb)
            plt.ylabel(self.axes[1])
            
            plt.gca().set_aspect(1)
            
        finalize_plot(title=title, show=show)
            

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
    


class UnitGrid(CartesianGridBase):
    r""" d-dimensional Cartesian grid with unit discretization in all directions 
    
    The grids can be thought of as a collection of d-dimensional cells of unit
    length. The `shape` parameter determines how many boxes there are in each
    direction. The cells are enumerated starting with 0, so the last cell has
    index :math:`n-1` if there are :math:`n` cells along a dimension. A given
    cell :math:`i` extends from coordinates :math:`i` to :math:`i + 1`, so the
    midpoint is at :math:`i + \frac12`, which is the cell coordinate. Taken
    together, the cells covers the interval :math:`[0, n]` along this dimension.
    """
    
    
    def __init__(self, shape, periodic: Union[List[bool], bool] = False):
        """ 
        Args:
            shape (tuple): The number of support points for each axis. The
                dimension of the grid is given by `len(shape)`.
            periodic (bool or list): Specifies which axes possess periodic
                boundary conditions. This is either a list of booleans defining
                periodicity for each individual axis or a single boolean value
                specifying the same periodicity for all axes.
        """
        super().__init__(shape, periodic)
        self.cuboid = Cuboid(np.zeros(self.dim), self.shape)
        self._discretization = np.ones(self.dim)
        
        # determine the cell center coordinates
        self._axes_coords = tuple(np.arange(n) + 0.5 for n in self.shape)
        self._axes_bounds = tuple(self.cuboid.bounds)


    @property
    def state(self) -> Dict[str, Any]:
        """ dict: the state of the grid """
        return {'shape': self.shape,
                'periodic': self.periodic}


    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "UnitGrid":  # type: ignore
        """ create a field from a stored `state`.
        
        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(shape=state_copy.pop('shape'),
                  periodic=state_copy.pop('periodic'))
        if state_copy:
            raise ValueError(f'State items {state_copy.keys()} were not used')
        return obj


    @property
    def volume(self) -> float:
        """ float: total volume of the grid """
        return float(np.prod(self.shape))


    def normalize_point(self, point, reduced_coords: bool = False):
        """ normalize coordinates by applying periodic boundary conditions
        
        Args:
            points (:class:`numpy.ndarray`): An array of coordinates
            reduced_coords (bool): Flag determining whether only the coordinates
                corresponding to axes in this grid are given
                 
        Returns:
            :class:`numpy.ndarray`: The respective coordinates with periodic
            boundary conditions applied.
        """
        point = np.asarray(point, dtype=np.double)
        if point.size == 0:
            return np.zeros((0, self.dim))
        if point.ndim == 0:
            if self.dim > 1:
                raise DimensionError(f'Dimension mismatch: Point {point} is '
                                     f'not of dimension {self.dim}.')
        elif point.shape[-1] != self.dim:
            raise DimensionError('Dimension mismatch: Array of shape '
                                 f'{point.shape} does not describe points of '
                                 f'dimension {self.dim}.')
        
        # normalize the coordinates for the periodic dimensions
        if point.ndim == 0:
            if self.periodic[0]:
                point %= self.shape[0]
        else:
            for i in range(self.num_axes):
                if self.periodic[i]:
                    point[..., i] %= self.shape[i]

        return point
    

    def cell_to_point(self, cells, cartesian: bool = True):
        """ convert cell coordinates to real coordinates
        
        Args:
            cells (:class:`numpy.ndarray`): Indices of the cells whose center
                coordinates are requested. This can be float values to indicate
                positions relative to the cell center.
            cartesian (bool): Determines whether the point is returned in
                Cartesian coordinates or grid coordinates. This does not have
                any effect for Cartesian coordinate systems, but the argument is
                retained to have a unified interface for all grids.
                
        Returns:
            :class:`numpy.ndarray`: The center points of the respective cells
        """
        cells = np.asanyarray(cells, dtype=np.double)
        if cells.size == 0:
            return np.zeros((0, self.dim))
        if cells.shape[-1] != self.dim:
            raise DimensionError('Dimension mismatch')
        return cells + 0.5


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
        return self.normalize_point(points).astype(np.int)
        
    
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
        diff = np.atleast_1d(p2) - np.atleast_1d(p1)
        # correct the periodic dimensions
        for i in range(self.num_axes):
            if self.periodic[i]:
                s = self.shape[i]
                diff[..., i] = (diff[..., i] + s/2) % s - s/2
        return diff
    
    
    def to_cartesian(self) -> "CartesianGrid":
        """ convert unit grid to CartesianGrid """
        return CartesianGrid(self.cuboid.bounds, shape=self.shape,
                             periodic=self.periodic)
        
        
    def get_subgrid(self, indices: Sequence[int]) -> "UnitGrid":
        """ return a subgrid of only the specified axes
        
        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid
                
        Returns:
            :class:`UnitGrid`: The subgrid
        """
        subgrid = self.__class__(shape=[self.shape[i] for i in indices],
                                 periodic=[self.periodic[i] for i in indices])
        subgrid.axes = [self.axes[i] for i in indices]
        return subgrid



class CartesianGrid(CartesianGridBase):
    r""" d-dimensional Cartesian grid with uniform discretization for each axis
    
    The grids can be thought of as a collection of n-dimensional boxes, called
    cells, of equal length in each dimension. The bounds then defined the total
    volume covered by these cells, while the cell coordinates give the location
    of the box centers. We index the boxes starting from 0 along each dimension.
    Consequently, the cell :math:`i-\frac12` corresponds to the left edge of the 
    covered interval and the index :math:`i+\frac12` corresponds to the right
    edge, when the dimension is covered by d boxes.
    
    In particular, the discretization along dimension :math:`k` is defined as

    .. math::
            x^{(k)}_i &= x^{(k)}_\mathrm{min} + \left(i + \frac12\right)
                \Delta x^{(k)}
            \quad \text{for} \quad i = 0, \ldots, N^{(k)} - 1
        \\
            \Delta x^{(k)} &= \frac{x^{(k)}_\mathrm{max} -
                                    x^{(k)}_\mathrm{min}}{N^{(k)}}
                                    
    where :math:`N^{(k)}` is the number of cells along this dimension.
    Consequently, the cells have dimension :math:`\Delta x^{(k)}` and cover the
    interval :math:`[x^{(k)}_\mathrm{min}, x^{(k)}_\mathrm{max}]`.
    """
    
    
    def __init__(self, bounds, shape,
                 periodic: Union[List[bool], bool] = False):
        """ 
        Args:
            bounds (tuple): Give the coordinate range for each axis. This should
                be a tuple of two number (lower and upper bound) for each axis.
                The length of `bounds` thus determines the grid dimension. 
            shape (tuple): The number of support points for each axis. The
                length of `shape` needs to match the grid dimension. 
            periodic (bool or list): Specifies which axes possess periodic
                boundary conditions. This is either a list of booleans defining
                periodicity for each individual axis or a single boolean value
                specifying the same periodicity for all axes.
        """
        bounds = np.array(bounds, ndmin=1, dtype=np.double)
        if bounds.shape == (2,):
            raise ValueError('`bounds with shape (2,) are ambiguous. Either '
                             'use shape (1, 2) to set up a 1d system with two '
                             'bounds or shape (2, 1) for a 2d system with only '
                             'the upper bounds specified') 
        
        if bounds.ndim == 1 or bounds.shape[1] == 1:
            # only set the upper bounds
            bounds = np.atleast_1d(np.squeeze(bounds))
            self.cuboid = Cuboid(np.zeros_like(bounds), bounds, mutable=False)
            
        elif bounds.ndim == 2 and bounds.shape[1] == 2:
            # upper and lower bounds of the grid are given
            self.cuboid = Cuboid.from_bounds(bounds, mutable=False)
            
        else:
            raise ValueError('Do not know how to interpret shape '
                             f'{bounds.shape} for bounds')
            
        # handle the shape array
        shape = _check_shape(shape)
        if len(shape) == 1 and self.cuboid.dim > 1:
            shape = np.full(self.cuboid.dim, shape, dtype=np.uint32)
        if self.cuboid.dim != len(shape):
            raise DimensionError('Dimension of the bounds and the shape are '
                                 'not compatible')
                    
        # initialize the base class            
        super().__init__(shape, periodic)
            
        # determine the coordinates
        p1, p2 = self.cuboid.corners
        axes_coords, discretization = [], []
        for d in range(self.dim):
            c, dc = np.linspace(p1[d], p2[d], self.shape[d],
                                endpoint=False, retstep=True)
            if self.shape[d] == 1:
                # correct for singular dimension
                dc = p2[d] - p1[d]
            c += dc / 2
            axes_coords.append(c)
            discretization.append(dc)
        self._discretization = np.array(discretization)
        self._axes_coords = tuple(axes_coords)
        self._axes_bounds = tuple(self.cuboid.bounds)

    
    @property
    def state(self) -> Dict[str, Any]:
        """ dict: the state of the grid """
        return {'bounds': self.axes_bounds,
                'shape': self.shape,
                'periodic': self.periodic}


    @classmethod
    def from_state(cls,  # type: ignore
                   state: Dict[str, Any]) -> "CartesianGrid":
        """ create a field from a stored `state`.
        
        Args:
            state (dict):
                The state from which the grid is reconstructed.
        """
        state_copy = state.copy()
        obj = cls(bounds=state_copy.pop('bounds'),
                  shape=state_copy.pop('shape'),
                  periodic=state_copy.pop('periodic'))
        if state_copy:
            raise ValueError(f'State items {state_copy.keys()} were not used')
        return obj

                    
    @property
    def volume(self) -> float:
        """ float: total volume of the grid """
        return float(self.cuboid.volume)


    def normalize_point(self, point, reduced_coords: bool = False):
        """ normalize coordinates by applying periodic boundary conditions
        
        Args:
            points (:class:`numpy.ndarray`): An array of coordinates
            reduced_coords (bool): Flag determining whether only the coordinates
                corresponding to axes in this grid are given
                 
        Returns:
            :class:`numpy.ndarray`: The respective coordinates with periodic
            boundary conditions applied.
        """
        point = np.asarray(point, dtype=np.double)
        if point.size == 0:
            return np.zeros((0, self.dim)) 
        if point.ndim == 0:
            if self.dim > 1:
                raise DimensionError(f'Dimension mismatch: Point {point} is '
                                     f'not of dimension {self.dim}.')
        elif point.shape[-1] != self.dim:
            raise DimensionError('Dimension mismatch: Array of shape '
                                 f'{point.shape} does not describe points of '
                                 f'dimension {self.dim}.')
        
        if any(self.periodic):
            if self.dim == 1:  # single, periodic dimension
                offset = self.cuboid.pos
                point = (point - offset) % self.cuboid.size + offset
            else:  # multiple dimensions => extract the periodic ones
                offset = self.cuboid.pos[self.periodic]
                size = self.cuboid.size[self.periodic]
                point[..., self.periodic] = \
                            (point[..., self.periodic] - offset) % size + offset
        return point


    def cell_to_point(self, cells, cartesian: bool = True):
        """ convert cell coordinates to real coordinates
        
        Args:
            cells (:class:`numpy.ndarray`): Indices of the cells whose center
                coordinates are requested. This can be float values to indicate
                positions relative to the cell center.
            cartesian (bool): Determines whether the point is returned in
                Cartesian coordinates or grid coordinates. This does not have
                any effect for Cartesian coordinate systems, but the argument is
                retained to have a unified interface for all grids.
                
        Returns:
            :class:`numpy.ndarray`: The center points of the respective cells
        """
        cells = np.atleast_1d(cells)
        if cells.size == 0:
            return cells
        else:
            return self.cuboid.pos + (cells + 0.5) * self.discretization


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
        points = self.normalize_point(points)
        cell_coords = (points - self.cuboid.pos) / self.discretization
        return cell_coords.astype(np.int)

        
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
        if any(self.periodic):
            size = self.cuboid.size[self.periodic]
            diff[..., self.periodic] = \
                            (diff[..., self.periodic] + size/2) % size - size/2
        return diff


    def get_subgrid(self, indices: Sequence[int]) -> "CartesianGrid":
        """ return a subgrid of only the specified axes
        
        Args:
            indices (list):
                Indices indicating the axes that are retained in the subgrid
                
        Returns:
            :class:`CartesianGrid`: The subgrid
        """
        subgrid = self.__class__(bounds=[self.axes_bounds[i] for i in indices],
                                 shape=[self.shape[i] for i in indices],
                                 periodic=[self.periodic[i] for i in indices])
        subgrid.axes = [self.axes[i] for i in indices]
        return subgrid

