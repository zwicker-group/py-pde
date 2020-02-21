'''
Bases classes

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
 
'''

import json
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import (List, Tuple, Dict, Any, Union, Callable, Generator,
                    TYPE_CHECKING)

import numpy as np

from ..tools.numba import jit
from ..tools.cache import cached_property, cached_method



if TYPE_CHECKING:
    from .boundaries.axes import Boundaries  # @UnusedImport



PI_4 = 4 * np.pi
PI_43 = 4 / 3 * np.pi



def _check_shape(shape) -> Tuple[int, ...]:
    """ checks the consistency of shape tuples """
    if not hasattr(shape, '__iter__'):
        shape = [shape]  # support single numbers
    result = []
    for dim in shape:
        if dim == int(dim) and dim >= 1:
            result.append(int(dim))
        else:
            raise ValueError(f'{repr(dim)} is not a valid number of support '
                             'points')
    return tuple(result) 




def discretize_interval(x_min: float, x_max: float, num: int) \
        -> Tuple[np.ndarray, float]:
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




class GridBase(metaclass=ABCMeta):
    """ Base class for all grids defining common methods and interfaces
    
    Attributes:
        dim (int):
            The spatial dimension in which the grid is embedded
        shape (tuple):
            The number of support points in each axis. Note that `len(shape)`
            might be smaller than `dim` if the grid assumes some symmetry. For
            instance, a spherically symmetric grid has `dim == 3`, but
            `len(shape) == 1`.
        discretization (:class:`numpy.ndarray`):
            The discretization along each axis.
        axes (tuple):
            The name of all the axes that are described by the grid.
        axes_symmetric (tuple):
            The names of the additional axes that the fields do not depend on,
            e.g. along which they are constant.
        axes_coords (tuple):
            The discretization points along each axis
        axes_bounds (tuple):
            The coordinate bounds of each axes
    """
    
    _subclasses: Dict[str, Any] = {}  # all classes inheriting from this
    coordinate_constraints: List[int] = []  # axes not described explicitly
    axes_symmetric: List[str] = []
    
    # defaults for properties that are defined in subclasses
    dim: int
    shape: Tuple[int, ...]
    axes: List[str]
    num_axes: int
    discretization: Any
    cell_volume_data: float
    axes_coords: Tuple
    axes_bounds: Tuple[Tuple[float, float], ...]
    periodic: List[bool]


    def __init__(self):
        """ initialize the grid """
        self._logger = logging.getLogger(self.__class__.__module__)


    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls


    @classmethod
    def from_state(cls, state: Union[str, Dict[str, Any]]) -> "GridBase":
        """ create a field from a stored `state`.
        
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
        class_name = state.pop('class')
        if class_name == cls.__name__:
            raise RuntimeError('Cannot reconstruct abstract class '
                               f'`{class_name}`')
        grid_cls = cls._subclasses[class_name]
        return grid_cls.from_state(state)  # type: ignore
    
    
    @abstractproperty
    def state(self) -> Dict[str, Any]: pass


    @property
    def state_serialized(self) -> str:
        """ str: JSON-serialized version of the state of this grid """
        state = self.state
        state['class'] = self.__class__.__name__
        return json.dumps(state)
    

    def copy(self) -> 'GridBase':
        """ return a copy of the grid """
        return self.__class__.from_state(self.state)


    def __repr__(self):
        """ return instance as string """        
        args = ', '.join(str(k) + '=' + str(v) for k, v in self.state.items())
        return f'{self.__class__.__name__}({args})'
         
         
    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.shape == other.shape and
                self.axes_bounds == other.axes_bounds and
                self.periodic == other.periodic)
        
        
    def __ne__(self, other):
        return (self.__class__ != other.__class__ or
                self.shape != other.shape or
                self.axes_bounds != other.axes_bounds or
                self.periodic != other.periodic)
        
         
    def compatible_with(self, other) -> bool:
        """ tests whether this class is compatible with other grids.
        
        Grids are compatible when they cover the same area with the same
        discretization. The difference to equality is that compatible grids do
        not need to have the same periodicity in their boundaries.
        
        Args:
            other (GridBase): The other grid to test against
        """
        return (self.__class__ == other.__class__ and  # type: ignore
                self.shape == other.shape and
                self.axes_bounds == other.axes_bounds)
         
         
    def assert_grid_compatible(self, other):
        """ checks whether `other` is compatible with the current grid
        
        Args:
            other (:class:`~pde.grids.GridBase`):
                The grid compared to this one
        
        Raises:
            ValueError: if grids are not compatible
        """
        if not self.compatible_with(other):
            raise ValueError('Grids incompatible')
        
        
    @property
    def numba_type(self) -> str:
        """ str: represents type of the grid data in numba signatures """
        return "f8[" + ', '.join([':'] * self.num_axes) + "]"
    
        
    @cached_property()
    def cell_coords(self):
        """ :class:`numpy.ndarray`: the coordinates of each cell """
        return np.moveaxis(np.meshgrid(*self.axes_coords, indexing='ij'), 0, -1)
        
    
    @cached_property()
    def cell_volumes(self):
        """ :class:`numpy.ndarray`: volume of each cell """
        return np.broadcast_to(self.cell_volume_data, self.shape)
        
        
    def distance_real(self, p1, p2) -> float:
        """Calculate the distance between two points given in real coordinates
        
        This takes periodic boundary conditions into account if need be
        
        Args:
            p1 (vector): First position
            p2 (vector): Second position
            
        Returns:
            float: Distance between the two positions
        """
        diff = self.difference_vector_real(p1, p2)
        return np.linalg.norm(diff, axis=-1)  # type: ignore


    @abstractproperty
    def volume(self) -> float: pass
    @abstractmethod
    def normalize_point(self, point, reduced_coords: bool = False): pass
    @abstractmethod
    def cell_to_point(self, cells, cartesian: bool = True): pass
    @abstractmethod
    def point_to_cell(self, points): pass
    @abstractmethod
    def point_to_cartesian(self, points): pass
    @abstractmethod
    def point_from_cartesian(self, points): pass
    @abstractmethod
    def difference_vector_real(self, p1, p2): pass
    @abstractmethod
    def polar_coordinates_real(self, origin, ret_angle=False): pass
    @abstractmethod
    def contains_point(self, point): pass
    @abstractmethod
    def iter_mirror_points(self, point, with_self: bool = False,
                           only_periodic: bool = True) -> Generator: pass
                           
    @abstractmethod
    def get_boundary_conditions(self, bc='natural') -> "Boundaries": pass
    @abstractmethod
    def get_operator(self, op, bc): pass
    @abstractmethod
    def get_line_data(self, data, extract: str = 'auto'): pass
    @abstractmethod
    def get_image_data(self, data): pass
    @abstractmethod
    def get_random_point(self, boundary_distance: float = 0,
                         cartesian: bool = True): pass
             
                         
    def plot(self):
        """ visualize the grid """
        raise NotImplementedError('Plotting is not implemented for class '
                                  f'{self.__class__.__name__}')
    

    @property
    def typical_discretization(self) -> float:
        """ float: the average side length of the cells """
        return np.mean(self.discretization)  # type: ignore

    
    def integrate(self, data) -> float:
        """ Integrates the discretized data over the grid
        
        Args:
            data (array): The values at the support points of the grid that 
                need to be integrated
        
        Returns:
            float: The values integrated over the entire grid
        """
        data = np.broadcast_to(data, self.shape)
        return float((data * self.cell_volume_data).sum())
    
    
    @cached_method()
    def make_normalize_point_compiled(self) -> Callable:
        """ return a compiled function that normalizes the points

        Normalizing points is useful to respect periodic boundary conditions. 
        Here, points are assumed to be specified by the physical values along
        the non-symmetric axes of the grid.
        """
        periodic_axes = np.flatnonzero(self.periodic)
        bounds = np.array(self.axes_bounds)
        offset = bounds[:, 0]
        size = bounds[:, 1] - bounds[:, 0]
        
        @jit
        def normalize_point(point):
            for i in periodic_axes:
                point[i] = (point[i] - offset[i]) % size[i] + offset[i]

        return normalize_point  # type: ignore
    
    
    @cached_method()
    def make_cell_volume_compiled(self, flat_index: bool = False) -> Callable:
        """ return a compiled function returning the volume of a grid cell
        
        Args:
            flat_index (bool): When True, cell_volumes are indexed by a single
                integer into the flattened array.
                
        Returns:
            function: returning the volume of the chosen cell
        """
        if np.isscalar(self.cell_volume_data):
            cell_volume_data = self.cell_volume_data
            @jit
            def cell_volume(*args) -> float:
                return cell_volume_data
        else:
            cell_volumes = self.cell_volumes
            
            if flat_index:
                @jit
                def cell_volume(idx: int) -> float:
                    return cell_volumes.flat[idx]  # type: ignore
            else:
                @jit
                def cell_volume(*args) -> float:
                    return cell_volumes[args]  # type: ignore
            
        return cell_volume  # type: ignore
    
    
    def make_interpolator_compiled(self, method: str = 'linear', bc='natural') \
            -> Callable:
        """ return a compiled function for interpolating values on the grid
        
        This interpolator respects boundary conditions and can thus interpolate
        values in the whole grid volume. However, close to corners, the
        interpolation might not be optimal, in particular for periodic grids.
        
        Args:
            method (str): Determines how the interpolation is done. Currently,
                only linear interpolation is supported.
            bc: Sets the boundary condition, which affects how values at the
                boundary are determined
                
        Returns:
            A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. The signature of
            this function is (data, point), where `data` is the numpy array
            containing the field data and position is denotes the position in
            grid coordinates.
        """
        if method != 'linear':
            raise ValueError(f"Unsupported interpolation method: '{method}'")

        bcs = self.get_boundary_conditions(bc)
        
        if self.num_axes == 1:
            # specialize for 1-dimensional interpolation
            lo = self.axes_bounds[0][0]
            dx = self.discretization[0]
            size = self.shape[0]
            normalize_point = self.make_normalize_point_compiled()
            ev = bcs[0].get_point_evaluator()
        
            @jit
            def interpolate_single(data, point):
                """ obtain interpolated value of data at a point
                
                Args:
                    data (:class:`numpy.ndarray`):  values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                
                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                normalize_point(point)
                c_l, d_l = divmod((point[0] - lo) / dx - 0.5, 1.)
                if c_l < -1 or c_l > size - 1:
                    raise ValueError('Point lies outside grid')
                c_li = int(c_l)
                c_hi = c_li + 1
                return (1 - d_l) * ev(data, (c_li,)) + d_l * ev(data, (c_hi,))  
            
        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            size_x, size_y = self.shape
            lo_x, lo_y = np.array(self.axes_bounds)[:, 0]
            dx, dy = self.discretization
            periodic_x, periodic_y = self.periodic
            ev_x = bcs[0].get_point_evaluator()
            ev_y = bcs[1].get_point_evaluator()
        
            @jit
            def interpolate_single(data, point):
                """ obtain interpolated value of data at a point
                
                Args:
                    data (:class:`numpy.ndarray`): The values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                
                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)
                
                value = np.zeros(data.shape[:-2])
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
                    raise ValueError('Point lies outside grid')
                            
                return value / weight
            
        elif self.num_axes == 3:
            # specialize for 3-dimensional interpolation
            size_x, size_y, size_z = self.shape
            lo_x, lo_y, lo_z = np.array(self.axes_bounds)[:, 0]
            dx, dy, dz = self.discretization
            periodic_x, periodic_y, periodic_z = self.periodic
            ev_x = bcs[0].get_point_evaluator()
            ev_y = bcs[1].get_point_evaluator()
            ev_z = bcs[2].get_point_evaluator()
        
            @jit
            def interpolate_single(data, point):
                """ obtain interpolated value of data at a point
                
                Args:
                    data (:class:`numpy.ndarray`): The values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                
                Returns:
                    :class:`numpy.ndarray`: The interpolated value at the point
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.)
                c_lz, d_lz = divmod((point[2] - lo_z) / dz - 0.5, 1.)
                w_x = (1 - d_lx, d_lx)
                w_y = (1 - d_ly, d_ly)
                w_z = (1 - d_lz, d_lz)
                
                value = np.zeros(data.shape[:-3])
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
                    raise ValueError('Point lies outside grid')
                            
                return value / weight            
            
        else:
            raise NotImplementedError('Compiled interpolation not implemented '
                                      f'for dimension {self.num_axes}')
            
        return interpolate_single  # type: ignore
                

    def make_add_interpolated_compiled(self) -> Callable:
        """ return a compiled function to add amounts at interpolated positions
                
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
            def add_interpolated(data, point, amount):
                """ add an amount to a field at an interpolated position 
                
                Args:
                    data (:class:`numpy.ndarray`): The values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                    amount (float or :class:`numpy.ndarray`): The amount that
                        will be added to the data. This value describes an
                        integrated quantity (given by the field value times the
                        discretization volume). This is important for
                        consistency with different discretizations and in
                        particular grids with non-uniform discretizations.
                """
                c_l, d_l = divmod((point[0] - lo) / dx - 0.5, 1.)
                if c_l < -1 or c_l > size - 1:
                    raise ValueError('Point lies outside grid')
                c_li = int(c_l)
                c_hi = c_li + 1
                
                if periodic:
                    c_li %= size
                    c_hi %= size
                    w_l = 1 - d_l  # weights of the low point
                    w_h = d_l      # weights of the high point
                    data[..., c_li] += w_l * amount / cell_volume(c_li)
                    data[..., c_hi] += w_h * amount / cell_volume(c_hi)

                elif c_li < 0:
                    if c_hi >= size:
                        raise RuntimeError('Point lies outside grid')
                    else:  # c_hi < size
                        data[..., c_hi] += amount / cell_volume(c_hi)
                else:  # c_li >= 0
                    if c_hi >= size:
                        data[..., c_li] += amount / cell_volume(c_li)
                    else:  # c_hi < size
                        w_l = 1 - d_l  # weights of the low point
                        w_h = d_l      # weights of the high point
                        data[..., c_li] += w_l * amount / cell_volume(c_li)
                        data[..., c_hi] += w_h * amount / cell_volume(c_hi)
            
        elif self.num_axes == 2:
            # specialize for 2-dimensional interpolation
            size_x, size_y = self.shape
            lo_x, lo_y = np.array(self.axes_bounds)[:, 0]
            dx, dy = self.discretization
            periodic_x, periodic_y = self.periodic

            @jit
            def add_interpolated(data, point, amount):
                """ add an amount to a field at an interpolated position 
                
                Args:
                    data (:class:`numpy.ndarray`): The values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                    amount (float or :class:`numpy.ndarray`): The amount that
                        will be added to the data. This value describes an
                        integrated quantity (given by the field value times the
                        discretization volume). This is important for
                        consistency with different discretizations and in
                        particular grids with non-uniform discretizations.
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.)
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
                    raise ValueError('Point lies outside grid')
        
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
            def add_interpolated(data, point, amount):
                """ add an amount to a field at an interpolated position 
                
                Args:
                    data (:class:`numpy.ndarray`): The values at the grid points
                    point (:class:`numpy.ndarray`): Coordinates of a single
                        point in the grid coordinate system
                    amount (float or :class:`numpy.ndarray`): The amount that
                        will be added to the data. This value describes an
                        integrated quantity (given by the field value times the
                        discretization volume). This is important for
                        consistency with different discretizations and in
                        particular grids with non-uniform discretizations.
                """
                # determine surrounding points and their weights
                c_lx, d_lx = divmod((point[0] - lo_x) / dx - 0.5, 1.)
                c_ly, d_ly = divmod((point[1] - lo_y) / dy - 0.5, 1.)
                c_lz, d_lz = divmod((point[2] - lo_z) / dz - 0.5, 1.)
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
                    raise ValueError('Point lies outside grid')
        
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
            raise NotImplementedError('Compiled interpolation not implemented '
                                      f'for dimension {self.num_axes}')
            
        return add_interpolated  # type: ignore
                

