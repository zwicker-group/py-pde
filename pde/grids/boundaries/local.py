r"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>

This module contains classes for handling a single boundary of a non-periodic
axis. Since an axis has two boundary, we simply distinguish them by a boolean
flag `upper`, which is True for the side of the axis with the larger coordinate.

The module currently supports different boundary conditions:

* :class:`~pde.grids.boundaries.local.DirichletBC`:
  Imposing the value of a field at the boundary
* :class:`~pde.grids.boundaries.local.NeumannBC`:
  Imposing the derivative of a field in the outward normal direction at the
  boundary
* :class:`~pde.grids.boundaries.local.MixedBC`:
  Imposing the derivative of a field in the outward normal direction
  proportional to its value at the boundary  
* :class:`~pde.grids.boundaries.local.CurvatureBC`:
  Imposing the second derivative (curvature) of a field at the boundary
* :class:`~pde.grids.boundaries.local.ExtrapolateBC`:
  Extrapolate boundary points linearly from the two points closest to the
  boundary

Derivatives are given in the direction of the outward normal vector, such that
positive derivatives correspond to a function that increases across the
boundary, which corresponds to an inwards flux. Conversely, negative
derivatives are associated with effluxes.
"""

import logging
import numbers
from abc import ABCMeta, abstractmethod
from typing import Any, Union, Tuple, Dict, Sequence, Optional, Callable, List
    
import numba as nb
import numpy as np
from numba.extending import register_jitable

from ..base import GridBase
from ...tools.docstrings import fill_in_docstring
from ...tools.numba import address_as_void_pointer



VectorType = Optional[Sequence[float]]
TensorType = Optional[Sequence[Sequence[float]]]
BoundaryData = Union[Dict, str, "BCBase"]



def _get_arr_1d(arr, idx: Tuple[int, ...], axis: int) \
        -> Tuple[np.ndarray, int, Tuple]:
    """ extract the 1d array along axis at point idx
    
    Args:
        arr (:class:`numpy.ndarray`): The full data array
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
        bc_idx: Tuple = (...,)
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



def _make_get_arr_1d(dim: int, axis: int) -> Callable:
    """ create function that extracts a 1d array at a given position
    
    Args:
        dim (int): The dimension of the space, i.e., the number of axes in the
            supplied data array
        axis (int): The axis that is returned as the 1d array
        
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
    
    @register_jitable
    def get_arr_1d(arr, idx: Tuple[int, ...]) -> Tuple[np.ndarray, int, Tuple]:
        """ extract the 1d array along axis at point idx """
        # extract the correct indices
        if dim == 1:
            i = idx[0]
            bc_idx: Tuple = (...,)
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
        
    return get_arr_1d  # type: ignore



class BCBase(metaclass=ABCMeta):
    """ represents a single boundary in an BoundaryPair instance """
    
    names: List[str]
    """ list: identifiers used to specify the given boundary class """
    homogeneous: bool
    """ bool: determines whether the boundary condition depends on space """

    value_is_linked: bool 
    """ bool: flag that indicates whether the value associated with this
    boundary condition is linked to :class:`~numpy.ndarray` managed by external
    code. """ 

    _subclasses: Dict[str, 'BCBase'] = {}  # all classes inheriting from this
    _conditions: Dict[str, 'BCBase'] = {}  # mapping from all names to classes
    

    _value: Union[float, np.ndarray]
    

    @fill_in_docstring    
    def __init__(self, grid: GridBase,
                 axis: int,
                 upper: bool,
                 rank: int = 0,
                 value: Union[float, np.ndarray] = 0):
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
                Flag indicating whether this boundary condition is associated
                with the upper side of an axis or not. In essence, this
                determines the direction of the local normal vector of the
                boundary.
            rank (int):
                The tensorial rank of the value associated with the boundary
                condition.
            value (float or str or array):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition. If
                value is a single value (or tensor in case of tensorial boundary
                conditions), the same value is applied to all points.
                Inhomogeneous boundary conditions are possible by supplying an
                expression as a string, which then may depend on the axes names
                of the respective grid.
        """
        self.grid = grid
        self.axis = axis
        self.upper = upper
        self.rank = rank
        
        self._shape_tensor = (self.grid.dim,) * self.rank     
        self._shape_boundary = (self.grid.shape[:self.axis] +
                                self.grid.shape[self.axis + 1:])
        
        self.value = value  # type: ignore
        self._logger = logging.getLogger(self.__class__.__name__)
                

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
        if hasattr(cls, 'names'):
            for name in cls.names:
                cls._conditions[name] = cls

                
    @property
    def value(self) -> Union[float, np.ndarray]:
        return self._value
                
          
    @value.setter  # type: ignore
    @fill_in_docstring          
    def value(self, value: Union[float, np.ndarray] = 0):
        """ set the value of this boundary condition
        
        Warning:
            {WARNING_EXEC}
        
        Args:
            value (float or str or array):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition.
        """
        self._value_expression = value
        
        if isinstance(value, str):
            # inhomogeneous value given by an expression 
            if self.rank != 0:
                raise NotImplementedError('Expressions for boundary values are '
                                          'only supported for scalar values.')
            
            from ...tools.expressions import ScalarExpression
            self.homogeneous = False

            # determine which coordinates are allowed to vary            
            axes_ids = (list(range(self.axis)) +
                        list(range(self.axis + 1, self.grid.num_axes)))
            
            # parse the expression with the correct variables
            bc_vars = [self.grid.axes[i] for i in axes_ids]
            expr = ScalarExpression(value, bc_vars)

            # get the coordinates at each point of the boundary            
            bc_coords = np.meshgrid(*[self.grid.axes_coords[i]
                                      for i in axes_ids],
                                    indexing='ij')
            
            # determine the value at each of these points. Note that we here
            # iterate explicitly over all points because the expression might
            # not depend on some of the variables, but we still want the array
            # self._value to contain a value at each boundary point
            self._value = np.empty_like(bc_coords[0])
            coords = {name: 0 for name in bc_vars}
            for idx in np.ndindex(*self._value.shape):
                for i, name in enumerate(bc_vars):
                    coords[name] = bc_coords[i][idx]
                self._value[idx] = expr(**coords)
            
        elif np.isscalar(value):
            # homogeneous, scalar value
            self.homogeneous = True
            self._value = float(value)
            
        else:
            # assume tensorial and/or inhomogeneous values
            value = np.asarray(value, dtype=np.double)
            shape = self._shape_tensor + self._shape_boundary
            
            if value.ndim == 0:
                # value is a scalar, although np.isscalar(np.array(0)) == False
                self.homogeneous = True
                self._value = np.broadcast_to(value, self._shape_tensor)
                    
            elif value.shape == shape:
                # inhomogeneous field with all tensor components
                self.homogeneous = False
                self._value = value
            
            elif value.shape == self._shape_tensor:
                # homogeneous field with all tensor components
                self.homogeneous = True
                self._value = value
                    
            else:
                raise ValueError(f"Dimensions {value.shape} of the value are "
                                 f"incompatible with rank {self.rank} and "
                                 f"spatial dimensions {self._shape_boundary}.")
        self.value_is_linked = False
    
    
    def link_value(self, value: np.ndarray):
        """ link value of this boundary condition to external array """
        assert value.data.c_contiguous
        
        shape = self._shape_tensor + self._shape_boundary
        if value.shape != shape:
            raise ValueError(f"The shape of the value, {value.shape}, is "
                              "incompatible with the expected shape for this "
                             f"boundary condition, {shape}")
        self._value = value
        self.homogeneous = False
        self.value_is_linked = True
    
    
    def _make_value_getter(self) -> Callable:
        """ return a (compiled) function for obtaining the value.
        
        Note:
            This should only be used in numba compiled functions that need to
            support boundary values that can be changed after the function has
            been compiled. In essence, the helper function created here servers
            to get around the compile-time constants that are otherwise created.
            
        Warning:
            The returned function has a hard-coded reference to the memory
            address of the value error, which must thus be maintained in memory.
            If the address of self.value changes, a new function needs to be
            created by calling this factory function again.
        """
        
        value = self.value
        if not isinstance(value, np.ndarray):
            RuntimeError('self.value must be a `numpy.ndarray` for creating '
                         'a value getter.')
        
        # obtains memory address of array
        mem_addr = value.ctypes.data  # type: ignore
        
        @nb.jit(nb.typeof(value)(), inline='always')
        def get_value():
            """ helper function returning the linked array """
            return nb.carray(address_as_void_pointer(mem_addr),
                             value.shape, value.dtype)
        
        return get_value  # type: ignore
        

    @classmethod
    def get_help(cls) -> str:
        """ Return information on how boundary conditions can be set """
        types = ', '.join(f"'{subclass.names[0]}'"
                          for subclass in cls._subclasses.values()
                          if hasattr(subclass, 'names'))
        return (f"Possible types of boundary conditions are {types}. "
                "Values can be set using {'type': TYPE, 'value': VALUE}. "
                "Here, VALUE can be a scalar number, a vector for tensorial "
                "boundary conditions, or a string, which can be interpreted "
                "as a sympy expression. In the latter case, the names of the "
                "axes not associated with this boundary can be used as "
                "variables to describe inhomogeneous boundary conditions.")

    
    def __repr__(self):
        if self.value_is_linked:
            return (f"{self.__class__.__name__}("
                    f"axis={self.axis}, upper={self.upper}, "
                    f"value=<linked: {self.value.ctypes.data}>)")
        elif np.array_equal(self.value, 0):
            return (f"{self.__class__.__name__}("
                    f"axis={self.axis}, upper={self.upper}, rank={self.rank})")
        else:
            return (f"{self.__class__.__name__}("
                    f"axis={self.axis}, upper={self.upper}, "
                    f"value={self.value!r})")
    
    
    def __str__(self):
        if hasattr(self, 'names'):
            if np.array_equal(self.value, 0):
                return f'"{self.names[0]}"' 
            elif self.value_is_linked:
                return (f'{{"type": "{self.names[0]}", '
                        f'"value": <linked: {self.value.ctypes.data}>}}')
            else:
                return (f'{{"type": "{self.names[0]}", '
                        f'"value": {self.value}}}')
        else:
            return self.__repr__()
    
    
    def __eq__(self, other):
        """ checks for equality neglecting the `upper` property """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.__class__ == other.__class__ and
                self.grid == other.grid and 
                self.axis == other.axis and
                self.homogeneous == other.homogeneous and
                self.rank == other.rank and
                np.all(self.value == other.value))
        

    def __ne__(self, other):
        return not self == other
    
    
    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """ 
        if self.value_is_linked:
            return hash((self.__class__.__name__,
                         self.grid._cache_hash(),
                         self.axis))
        else:
            if isinstance(self.value, np.ndarray):
                value = self.value.tobytes()
            else:
                value = self.value
            return hash((self.__class__.__name__,
                         self.grid._cache_hash(),
                         self.axis,
                         value))


    def copy(self, upper: Optional[bool] = None,
             rank: int = None,
             value: Union[float, np.ndarray] = None) -> "BCBase":
        """ return a copy of itself, but with a reference to the same grid """
        obj = self.__class__(grid=self.grid,
                             axis=self.axis,
                             upper=self.upper if upper is None else upper,
                             rank=self.rank if rank is None else rank,
                             value=self.value if value is None else value)
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj
        
        
    @property
    def is_scalar(self) -> bool:
        """ bool: whether the boundary condition is a scalar """
        return self.rank == 0
    

    def extract_component(self, *indices):
        """ extracts the boundary conditions for the given component

        Args:
            *indices:
                One or two indices for vector or tensor fields, respectively
        """
        if not self.homogeneous:
            raise NotImplementedError('Only homogeneous boundary conditions '
                                      'support tensorial values')
            
        if np.array_equal(self.value, 0):  # special case for all ranks
            value = 0
            
        elif len(indices) == 0:  # scalar conditions
            value = self.value
            
        else:  # tensorial boundary conditions
            value = self.value[indices]
            
        rank = self.rank - len(indices)
        return self.copy(value=value, rank=rank)

                
    @classmethod
    def from_str(cls, grid: GridBase,
                 axis: int,
                 upper: bool,
                 condition: str,
                 rank: int = 0,
                 value=0, **kwargs) -> "BCBase":
        r""" creates boundary from a given string identifier
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            condition (str):
                Identifies the boundary condition
            rank (int):
                The tensorial rank of the value associated with the boundary
                condition.
            value (float or str or array):
                Sets the associated value
            \**kwargs:
                Additional arguments passed to the constructor
        """
        if condition == 'no-flux' and np.all(value == 0):
            condition = 'derivative'

        # extract the class
        try:
            boundary_class = cls._conditions[condition]
        except KeyError:
            raise ValueError(f'Boundary condition `{condition}` not defined. '
                             f'{cls.get_help()}')

        # create the actual class     
        return boundary_class(grid=grid, axis=axis, upper=upper,  # type: ignore
                              rank=rank, value=value, **kwargs)
        
        
    @classmethod
    def from_dict(cls, grid: GridBase,
                  axis: int,
                  upper: bool,
                  data: Dict[str, Any],
                  rank: int = 0) -> "BCBase":
        """ create boundary from data given in dictionary
         
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (dict):
                The dictionary defining the boundary condition
            rank (int):
                The tensorial rank of the value associated with the boundary
                condition.
        """
        data = data.copy()  # need to make a copy since we modify it below
        
        # parse all possible variants that could be given
        if data.keys() == {'value'}:
            # only a value is given => Assume Dirichlet conditions
            b_type = 'value'
            b_value = data.pop('value')
            
        elif data.keys() == {'derivative'}:
            # the derivative is obviously given => Assume Neumann conditions
            b_type = 'derivative'
            b_value = data.pop('derivative')
            
        elif data.keys() == {'mixed'}:
            # short notation for mixed condition
            b_type = 'mixed'
            b_value = data.pop('mixed')
            
        elif data.keys() == {'curvature'}:
            # short notation for curvature condition
            b_type = 'curvature'
            b_value = data.pop('curvature')
            
        elif 'type' in data.keys():
            # type is given (optionally with a value)
            b_type = data.pop('type')
            b_value = data.pop('value', 0)
            
        else:
            raise ValueError('Boundary condition defined by '
                             f'{str(list(data.keys()))} are not supported.')
        
        # initialize the boundary class with all remaining values forwarded
        return cls.from_str(grid, axis, upper, condition=b_type, rank=rank, 
                            value=b_value, **data)
        
        
    @classmethod
    def from_data(cls, grid: GridBase,
                  axis: int,
                  upper: bool,
                  data: BoundaryData,
                  rank: int = 0) -> "BCBase":
        """ create boundary from some data

        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (str or dict):
                Data that describes the boundary
            rank (int):
                The tensorial rank of the value associated with the boundary
                condition.
        
        Returns:
            :class:`~pde.grids.boundaries.local.BCBase`: the instance created
            from the data
            
        Throws:
            ValueError if `data` cannot be interpreted as a boundary condition
        """
        # check all different data formats
        if isinstance(data, BCBase):
            # already in the correct format
            assert data.grid == grid and data.axis == axis and data.rank == rank
            return data.copy(upper=upper)
        
        elif isinstance(data, dict):
            # create from dictionary
            return cls.from_dict(grid, axis, upper=upper, data=data, rank=rank)
        
        elif isinstance(data, str):
            # create a specific condition given by a string
            return cls.from_str(grid, axis, upper=upper, condition=data,
                                rank=rank)
        
        else:
            raise ValueError(f'Unsupported boundary format: `{data}`. '
                             f'{cls.get_help()}')


    def check_value_rank(self, rank: int):
        """ check whether the values at the boundaries have the correct rank
        
        Args:
            rank (tuple): The rank of the value that is stored with this
                boundary condition
            
        Throws:
            RuntimeError: if the value does not have rank `rank`
        """
        if self.rank != rank:
            raise RuntimeError(f"Expected rank {rank}, but boundary condition "
                               f"had rank {self.rank}.")
            
            
    @abstractmethod
    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        pass
            
    @abstractmethod
    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float: pass
            
    @abstractmethod
    def make_virtual_point_evaluator(self) -> Callable: pass
    
    @abstractmethod
    def make_adjacent_evaluator(self) -> Callable: pass
            
    @property
    def differentiated(self) -> "BCBase": 
        """ BCBase: differentiated version of this boundary condition """
        raise NotImplementedError



class BCBase1stOrder(BCBase):
    """ represents a single boundary in an BoundaryPair instance """


    @abstractmethod
    def get_virtual_point_data(self, compiled: bool = False) \
        -> Tuple: pass


    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        """ sets the elements of the sparse representation of this condition
        
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


    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        """ calculate the value of the virtual point outside the boundary 
        
        Args:
            arr (array): The data values associated with the grid
            idx (tuple): The index of the point to evaluate. This is a tuple of
                length `grid.num_axes` with the either -1 or `dim` as the entry
                for the axis associated with this boundary condition. Here,
                `dim` is the dimension of the axis. The index is optional if
                dim == 1.                 
            
        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                raise ValueError('Index `idx` can only be deduced for grids '
                                 'with a single axis.')
        
        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)
        
        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()
        
        if self.homogeneous:
            return (const + factor * arr_1d[..., index])  # type: ignore
        else:
            return (const[bc_idx] +  # type: ignore
                    factor[bc_idx] * arr_1d[..., index])


    def make_virtual_point_evaluator(self) -> Callable:
        """ returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """
        dx = self.grid.discretization[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        
        if not isinstance(dx, numbers.Number):
            raise ValueError(f'Discretization along axis {self.axis} must be a '
                             f'number, not `{dx}`')

        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data(compiled=True)
        
        if self.homogeneous:
            @register_jitable(inline='always')
            def virtual_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the virtual point at `idx` """
                arr_1d, _, _ = get_arr_1d(arr, idx)
                return (const() + factor() * arr_1d[..., index])  # type: ignore
                        
                
        else:
            @register_jitable(inline='always')
            def virtual_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the virtual point at `idx` """
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                return (const()[bc_idx] +  # type: ignore
                        factor()[bc_idx] * arr_1d[..., index])
        
        return virtual_point  # type: ignore
    
    
    def make_adjacent_evaluator(self) -> Callable:
        """ returns a function evaluating the value adjacent to a given point 

        Returns:
            function: A function with signature (arr_1d, i_point, bc_idx), where
            `arr_1d` is the one-dimensional data array (the data points along 
            the axis perpendicular to the boundary), `i_point` is the index into
            this array for the current point and bc_idx are the remaining
            indices of the current point, which indicate the location on the 
            boundary plane. The result of the function is the data value at the
            adjacent point along the axis associated with this boundary
            condition in the upper (lower) direction when `upper` is True
            (False).
        """
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

            @register_jitable(inline='always')
            def adjacent_point(arr_1d, i_point, bc_idx) -> float:
                """ evaluate the value adjacent to the current point """
                # determine the parameters for evaluating adjacent point. Note
                # that defining the variables c and f for the interior points
                # seems needless, but it turns out that this results in a 10x
                # faster function (because of branch prediction?).
                if i_point == i_bndry:
                    c, f, i = const(), factor(), index
                else:
                    c, f, i = 0., 1., i_point + i_dx  # INTENTIONAL; see above
                
                # calculate the values
                return c + f * arr_1d[..., i]  # type: ignore
                
        else:
            # the boundary condition is a function of space

            # calculate necessary constants
            const, factor, index = self.get_virtual_point_data(compiled=True)
            
            shape = self._shape_tensor + self._shape_boundary
            zeros, ones = np.zeros(shape), np.ones(shape)
            
            @register_jitable(inline='always')
            def adjacent_point(arr_1d, i_point, bc_idx) -> float:
                """ evaluate the value adjacent to the current point """
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
    


class DirichletBC(BCBase1stOrder):
    """ represents a boundary condition imposing the value """
    
    names = ['value', 'dirichlet']  # identifiers for this boundary condition

    
    def get_virtual_point_data(self, compiled: bool = False) -> Tuple:
        """ return data suitable for calculating virtual points
        
        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.
        
        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """        
        const = 2 * self.value
        factor = -1 if np.isscalar(const) else -np.ones_like(const)
        index = self.grid.shape[self.axis] - 1 if self.upper else 0
            
        if not compiled:
            return (const, factor, index)
        else:
            # return boundary data such that dynamically calculated values can
            # be used in numba compiled code. This is a work-around since numpy
            # arrays are copied into closures, making them compile-time
            # constants
            
            if self.value_is_linked:
                value = self._make_value_getter()
                
                @register_jitable(inline='always')
                def const_func():
                    return 2 * value()
            else:
                @register_jitable(inline='always')
                def const_func():
                    return const
            
            @register_jitable(inline='always')
            def factor_func():
                return factor
                
            return (const_func, factor_func, index)
    
            
    @property
    def differentiated(self) -> BCBase:
        """ BCBase: differentiated version of this boundary condition """
        return NeumannBC(grid=self.grid, axis=self.axis, upper=self.upper,
                         rank=self.rank, value=np.zeros_like(self.value))



class NeumannBC(BCBase1stOrder):
    """ represents a boundary condition imposing the derivative in the outward
    normal direction of the boundary """
    
    names = ['derivative', 'neumann']  # identifiers for this boundary condition        

    
    def get_virtual_point_data(self, compiled: bool = False) -> Tuple:
        """ return data suitable for calculating virtual points
        
        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.
                    
        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """        
        dx = self.grid.discretization[self.axis]
        
        const = dx * self.value
        factor = 1 if np.isscalar(const) else np.ones_like(const)
        index = self.grid.shape[self.axis] - 1 if self.upper else 0
            
        if not compiled:
            return (const, factor, index)
        else:
            # return boundary data such that dynamically calculated values can
            # be used in numba compiled code. This is a work-around since numpy
            # arrays are copied into closures, making them compile-time
            # constants
            
            if self.value_is_linked:
                value = self._make_value_getter()
                @register_jitable(inline='always')
                def const_func():
                    return dx * value()
            else:
                @register_jitable(inline='always')
                def const_func():
                    return const
            
            @register_jitable(inline='always')
            def factor_func():
                return factor
                
            return (const_func, factor_func, index)


    @property
    def differentiated(self) -> BCBase:
        """ BCBase: differentiated version of this boundary condition """
        return CurvatureBC(grid=self.grid, axis=self.axis, upper=self.upper,
                           rank=self.rank, value=np.zeros_like(self.value))



class MixedBC(BCBase1stOrder):
    r""" represents a mixed (or Robin) boundary condition imposing a derivative
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
    
    names = ['mixed', 'robin']
    
    def __init__(self, grid: GridBase,
                 axis: int,
                 upper: bool,
                 rank: int = 0,
                 value=0,
                 const: float = 0):
        r""" 
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated
                with the upper side of an axis or not. In essence, this
                determines the direction of the local normal vector of the
                boundary.
            rank (int):
                The tensorial rank of the value associated with the boundary
                condition.
            value (float or str or array):
                The parameter :math:`\gamma` quantifying the influence of the
                field onto its normal derivative. If `value` is a single value
                (or tensor in case of tensorial boundary conditions), the same
                value is applied to all points.  Inhomogeneous boundary
                conditions are possible by supplying an expression as a string,
                which then may depend on the axes names of the respective grid.
            const (float):
                The parameter :math:`\beta` determining the constant term for 
                the boundary condition. This term does not yet allow for
                tensorial or spatially varying values.
        """
        super().__init__(grid, axis, upper, rank, value)
        # TODO: support spatially varying constant terms β
        self.const = float(const)  
        
    def __eq__(self, other):
        """ checks for equality neglecting the `upper` property """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.const == other.const


    def _cache_hash(self) -> int:
        """ returns a value to determine when a cache needs to be updated """ 
        if isinstance(self.const, np.ndarray):
            const = self.const.tobytes()
        else:
            const = self.const  
        return hash((super()._cache_hash(), const))
    
    
    def copy(self, upper: Optional[bool] = None,
             rank: int = None,
             value: Union[float, np.ndarray] = None,
             const: Union[float, np.ndarray] = None) -> "MixedBC":
        """ return a copy of itself, but with a reference to the same grid """
        obj = self.__class__(grid=self.grid,
                             axis=self.axis,
                             upper=self.upper if upper is None else upper,
                             rank=self.rank if rank is None else rank,
                             value=self.value if value is None else value,
                             const=self.const if const is None else const)
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj
        
        
    def get_virtual_point_data(self, compiled: bool = False) -> Tuple:
        """ return data suitable for calculating virtual points
        
        Args:
            compiled (bool):
                Flag indicating whether a compiled version is required, which
                automatically takes updated values into account when it is used
                in numba-compiled code.
                    
        Returns:
            :class:`BC1stOrderData`: the data structure associated with this
            virtual point
        """        
        dx = self.grid.discretization[self.axis]
        
        factor = dx * self.value
        if np.isscalar(factor):
            if np.isinf(factor):
                const = 0
                factor = -1
            else:
                const = 2 * dx * self.const / (2 + factor)
                factor = (2 - factor) / (2 + factor)
        else:
            # calculate values assuming finite factor
            const = 2 * dx * self.const / (2 + factor)
            factor = (2 - factor) / (2 + factor)
            # correct at places of infinite values 
            const[np.isinf(factor)] = 0  # type: ignore
            factor[np.isinf(factor)] = -1
            
        index = self.grid.shape[self.axis] - 1 if self.upper else 0

        if not compiled:
            return (const, factor, index)
        
        # return boundary data such that dynamically calculated values can
        # be used in numba compiled code. This is a work-around since numpy
        # arrays are copied into closures, making them compile-time
        # constants

        if self.value_is_linked:
            const_val = float(self.const)  # only support scalar values
            value_func = self._make_value_getter()
            @register_jitable(inline='always')
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

            @register_jitable(inline='always')
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
            @register_jitable(inline='always')
            def const_func():
                return const          
            
            @register_jitable(inline='always')
            def factor_func():
                return factor
            
        return (const_func, factor_func, index)
    
    

class BCBase2ndOrder(BCBase):
    """ abstract base class for boundary conditions of 2nd order """
    
    
    @abstractmethod
    def get_virtual_point_data(self) -> Tuple[Any, Any, int, Any, int]:
        """ return data suitable for calculating virtual points
        
        Returns:
            tuple: the data associated with this virtual point 
        """ 


    def get_data(self, idx: Tuple[int, ...]) -> Tuple[float, Dict[int, float]]:
        """ sets the elements of the sparse representation of this condition
        
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

    
    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float:
        """ calculate the value of the virtual point outside the boundary 
        
        Args:
            arr (array): The data values associated with the grid
            idx (tuple): The index of the point to evaluate. This is a tuple of
                length `grid.num_axes` with the either -1 or `dim` as the entry
                for the axis associated with this boundary condition. Here,
                `dim` is the dimension of the axis. The index is optional if
                dim == 1.                 
            
        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                raise ValueError('Index `idx` can only be deduced for grids '
                                 'with a single axis.')

        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)
        
        # calculate necessary constants
        data = self.get_virtual_point_data()
        
        if self.homogeneous:
            return (data[0] +  # type: ignore
                    data[1] * arr_1d[..., data[2]] + 
                    data[3] * arr_1d[..., data[4]])
        else:
            return (data[0][bc_idx] +  # type: ignore
                    data[1][bc_idx] * arr_1d[..., data[2]] +
                    data[3][bc_idx] * arr_1d[..., data[4]])
    
        
    def make_virtual_point_evaluator(self) -> Callable:
        """ returns a function evaluating the value at the virtual support point

        Returns:
            function: A function that takes the data array and an index marking
            the current point, which is assumed to be a virtual point. The
            result is the data value at this point, which is calculated using
            the boundary condition.
        """
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        
        if size < 2:
            raise ValueError('Need at least two support points along axis '
                             f'{self.axis} to apply boundary conditions')
        if not isinstance(dx, numbers.Number):
            raise ValueError(f'Discretization along axis {self.axis} must be a '
                             f'number, not `{dx}`')

        # calculate necessary constants
        data = self.get_virtual_point_data()
        
        if self.homogeneous:
            @register_jitable
            def virtual_point(arr, idx: Tuple[int, ...]):
                """ evaluate the virtual point at `idx` """
                arr_1d, _, _ = get_arr_1d(arr, idx)
                
                return (data[0] +
                        data[1] * arr_1d[..., data[2]] +
                        data[3] * arr_1d[..., data[4]])
            
        else:
            @register_jitable
            def virtual_point(arr, idx: Tuple[int, ...]):
                """ evaluate the virtual point at `idx` """
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                
                return (data[0][bc_idx] +
                        data[1][bc_idx] * arr_1d[..., data[2]] +
                        data[3][bc_idx] * arr_1d[..., data[4]])
            
        return virtual_point  # type: ignore
        

    def make_adjacent_evaluator(self) -> Callable:
        """ returns a function evaluating the value adjacent to a given point 

        Returns:
            function: A function with signature (arr_1d, i_point, bc_idx), where
            `arr_1d` is the one-dimensional data array (the data points along 
            the axis perpendicular to the boundary), `i_point` is the index into
            this array for the current point and bc_idx are the remaining
            indices of the current point, which indicate the location on the 
            boundary plane. The result of the function is the data value at the
            adjacent point along the axis associated with this boundary
            condition in the upper (lower) direction when `upper` is True
            (False).
        """
        size = self.grid.shape[self.axis]
        if size < 2:
            raise ValueError('Need at least two support points along axis '
                             f'{self.axis} to apply boundary conditions')

        # get values distinguishing upper from lower boundary   
        if self.upper:
            i_bndry = size - 1
            i_dx = 1
        else:
            i_bndry = 0
            i_dx = -1

        # calculate necessary constants
        data_vp = self.get_virtual_point_data()
        
        if self.homogeneous:
            # the boundary condition does not depend on space
            if self.is_scalar:
                zeros = 0.
            else:
                zeros = np.zeros((self.grid.dim,) * self.rank)
            
            @register_jitable
            def adjacent_point(arr_1d, i_point, bc_idx):
                """ evaluate the value adjacent to the current point """
                # determine the parameters for evaluating adjacent point
                if i_point == i_bndry:
                    data = data_vp
                else:
                    data = (zeros, 1., i_point + i_dx, 0., 0)
                
                # calculate the values
                return (data[0] +
                        data[1] * arr_1d[..., data[2]] +
                        data[3] * arr_1d[..., data[4]])
                
        else:
            # the boundary condition is a function of space
            shape = tuple(self.grid.shape[i]
                          for i in range(self.grid.num_axes)
                          if i != self.axis)
            zeros, ones = np.zeros(shape), np.ones(shape)
            
            @register_jitable
            def adjacent_point(arr_1d, i_point, bc_idx):
                """ evaluate the value adjacent to the current point """
                # determine the parameters for evaluating adjacent point
                if i_point == i_bndry:
                    data = data_vp
                else:
                    data = (zeros, ones, i_point + i_dx, zeros, 0)
                
                return (data[0][bc_idx] +
                        data[1][bc_idx] * arr_1d[..., data[2]] +
                        data[3][bc_idx] * arr_1d[..., data[4]])
        
        return adjacent_point  # type: ignore    



class ExtrapolateBC(BCBase2ndOrder):
    """ represents a boundary condition that extrapolates the virtual point
    using two points close to the boundary
    
    This imposes a vanishing second derivative.
    """
    
    names = ['extrapolate', 'extrapolation']  # identifiers for this condition        

    
    def get_virtual_point_data(self) -> Tuple[float, float, int, float, int]:
        """ return data suitable for calculating virtual points
            
        Returns:
            tuple: the data structure associated with this virtual point
        """        
        size = self.grid.shape[self.axis]
        
        if size < 2:
            raise RuntimeError('Need at least 2 support points to use the '
                               'extrapolate boundary condition.')

        if self.upper:
            i1 = size - 1
            i2 = size - 2
        else:
            i1 = 0
            i2 = 1
        return (0., 2., i1, -1., i2)



class CurvatureBC(BCBase2ndOrder):
    """ represents a boundary condition imposing the 2nd derivative at the
    boundary """
    
    names = ['curvature', 'second_derivative']  # identifiers for this BC        

    
    def get_virtual_point_data(self) -> Tuple[Any, float, int, float, int]:
        """ return data suitable for calculating virtual points
            
        Returns:
            tuple: the data structure associated with this virtual point
        """        
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]
        
        if size < 2:
            raise RuntimeError('Need at least 2 support points to use the '
                               'curvature boundary condition.')

        value = self.value * dx**2
        ones = 1 if np.isscalar(value) else np.ones_like(value)
        if self.upper:
            f1, i1 = 2., size - 1
            f2, i2 = -1., size - 2
        else:
            f1, i1 = 2., 0
            f2, i2 = -1., 1
        return (value, f1 * ones, i1, f2 * ones, i2)
