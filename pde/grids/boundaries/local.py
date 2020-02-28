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

from abc import ABCMeta, abstractmethod
import numbers
from typing import Any, Union, Tuple, Dict, Sequence, Optional, Callable, List
    
import numpy as np

from ..base import GridBase
from ...tools.numba import jit
from ...tools.expressions import ScalarExpression



VectorType = Optional[Sequence[float]]
TensorType = Optional[Sequence[Sequence[float]]]
BoundaryData = Union[Dict, str, "BCBase"]



class DomainError(ValueError):
    """ exception indicating that point lies outside domain """ 
    pass



class PeriodicityError(RuntimeError):
    """ exception indicating that the grid periodicity is inconsistent """
    pass



def _get_arr_1d(arr, idx: Tuple[int, ...], axis: int) \
        -> Tuple[Any, int, Tuple[int, ...]]:
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
        bc_idx: Tuple[int, ...] = tuple()
        arr_1d = arr
        
    elif dim == 2:
        if axis == 0:
            i, y = idx
            bc_idx = (y,)
            arr_1d = arr[..., :, y]
        elif axis == 1:
            x, i = idx
            bc_idx = (x,)
            arr_1d = arr[..., x, :]
            
    elif dim == 3:
        if axis == 0:
            i, y, z = idx
            bc_idx = (y, z)
            arr_1d = arr[..., :, y, z]
        elif axis == 1:
            x, i, z = idx
            bc_idx = (x, z)
            arr_1d = arr[..., x, :, z]
        elif axis == 2:
            x, y, i = idx
            bc_idx = (x, y)
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
    
    @jit
    def get_arr_1d(arr, idx: Tuple[int, ...]) \
            -> Tuple[Any, int, Tuple[int, ...]]:
        """ extract the 1d array along axis at point idx """
        # extract the correct indices
        if dim == 1:
            i = idx[0]
            bc_idx: Tuple[int, ...] = tuple()
            arr_1d = arr
            
        elif dim == 2:
            if axis == 0:
                i, y = idx
                bc_idx = (y,)
                arr_1d = arr[..., :, y]
            elif axis == 1:
                x, i = idx
                bc_idx = (x,)
                arr_1d = arr[..., x, :]
                
        elif dim == 3:
            if axis == 0:
                i, y, z = idx
                bc_idx = (y, z)
                arr_1d = arr[..., :, y, z]
            elif axis == 1:
                x, i, z = idx
                bc_idx = (x, z)
                arr_1d = arr[..., x, :, z]
            elif axis == 2:
                x, y, i = idx
                bc_idx = (x, y)
                arr_1d = arr[..., x, y, :]
                
        else:
            raise NotImplementedError    
        
        return arr_1d, i, bc_idx
        
    return get_arr_1d  # type: ignore



class BCBase(metaclass=ABCMeta):
    """ represents a single boundary in an BoundaryPair instance """
    
    names: List[str]   # identifiers for the given boundary
    homogeneous: bool  # spatial dependence of boundary condition?
    value: Any         # the actual value of the boundary condition

    _subclasses: Dict[str, Any] = {}  # all classes inheriting from this
    _conditions: Dict[str, Any] = {}
    
    
    def __init__(self, grid: GridBase, axis: int, upper: bool, value=0):
        """ 
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
        self.set_value(value)
                
                    
    def set_value(self, value=0):
        """ set the value of this boundary condition
        
        Args:
            value (float or str or array):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition.
        """
        self._value_expression = value
        
        if isinstance(value, str):
            # inhomogeneous value given by an expression 
            self.homogeneous = False

            # determine which coordinates are allowed to vary            
            axes_ids = (list(range(self.axis)) +
                        list(range(self.axis + 1, self.grid.num_axes)))
            
            # parse the expression with the correct variables
            bc_vars = [self.grid.axes[i] for i in axes_ids]
            expr = ScalarExpression(value, bc_vars)
            
            # determine the value at each boundary position
            bc_coords = np.meshgrid(*[self.grid.axes_coords[i]
                                      for i in axes_ids],
                                    indexing='ij')
            self.value = expr(**{name: value
                                 for name, value in zip(bc_vars, bc_coords)})
            
        elif np.isscalar(value):
            # homogeneous, scalar value
            self.homogeneous = True
            self.value = float(value)
            
        else:
            # assume a homogeneous, tensorial value 
            self.homogeneous = True
            self.value = np.array(value, dtype=np.double)
            # check whether the value is actually a scalar. This is necessary
            # since np.isscalar(np.array(0)) == False
            if self.value.ndim == 0:
                self.value = float(self.value)
    

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
        if hasattr(cls, 'names'):
            for name in cls.names:
                cls._conditions[name] = cls


    @classmethod
    def get_help(cls) -> str:
        """ Return information on how boundary conditions can be set """
        types = ', '.join(f"'{subclass.names[0]}'"
                          for subclass in cls._subclasses.values()
                          if hasattr(subclass, 'names'))
        return (f"Possible types of boundary conditions are {types}. "
                "Values can be set using {'type': TYPE, 'value': VALUE}."
                "Here, VALUE can be a scalar number, a vector for tensorial "
                "boundary conditions, or a string, which can be interpreted "
                "as a sympy expression. In the latter case, the names of the "
                "axes not associated with this boundary can be used as "
                "variables to describe inhomogeneous boundary conditions.")

    
    def __repr__(self):
        if np.array_equal(self.value, 0):
            return (f"{self.__class__.__name__}("
                    f"axis={self.axis}, upper={self.upper}, "
                    f"homogeneous={self.homogeneous})")
        else:
            return (f"{self.__class__.__name__}("
                    f"axis={self.axis}, upper={self.upper}, "
                    f"homogeneous={self.homogeneous}, value={self.value!r})")
    
    
    def __str__(self):
        if hasattr(self, 'names'):
            if np.array_equal(self.value, 0):
                return f'"{self.names[0]}"' 
            elif self.homogeneous:
                return (f'{{"type": "{self.names[0]}", '
                        f'"value": {self.value}}}')
            else:
                return (f'{{"type": "{self.names[0]}", '
                        f'"value": "{self._value_expression}"}}')
        else:
            raise RuntimeError('This class should not be used directly')
    
    
    def __eq__(self, other):
        """ checks for equality neglecting the `upper` property """
        return (self.__class__ == other.__class__ and
                self.grid == other.grid and 
                self.axis == other.axis and
                self.homogeneous == other.homogeneous and
                np.all(self.value == other.value))
        

    def __neq__(self, other):
        return not self.__eq__(other)


    def copy(self, upper: Optional[bool] = None, value=None) -> "BCBase":
        """ return a copy of itself, but with a reference to the same grid """
        if upper is None:
            upper = self.upper
        if value is None:
            value = self._value_expression
        return self.__class__(grid=self.grid, axis=self.axis, upper=upper,
                              value=value)
        
        
    @property
    def rank(self) -> int:
        """ return the rank of the boundary condition """
        if self.homogeneous:
            if np.isscalar(self.value):
                return 0
            else:
                return self.value.ndim  # type: ignore
        else:
            return 0  # only scalar values by definition
        
        
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
            
        return self.copy(value=value)

                
    @classmethod
    def from_str(cls, grid: GridBase, axis: int, upper: bool, condition: str,
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
                              value=value, **kwargs)
        
        
    @classmethod
    def from_dict(cls, grid: GridBase, axis: int, upper: bool,
                  data: Dict[str, Any]) -> "BCBase":
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
        return cls.from_str(grid, axis, upper, condition=b_type, value=b_value,
                            **data)
        
        
    @classmethod
    def from_data(cls, grid: GridBase, axis: int, upper: bool,
                  data: BoundaryData) -> "BCBase":
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
        
        Returns:
            :class:`~pde.grids.boundaries.local.BCBase`: the instance created
            from the data
            
        Throws:
            ValueError if `data` cannot be interpreted as a boundary condition
        """
        # check all different data formats
        if isinstance(data, BCBase):
            # already in the correct format
            assert data.grid == grid and data.axis == axis
            return data.copy(upper=upper)
        
        elif isinstance(data, dict):
            # create from dictionary
            return cls.from_dict(grid, axis, upper=upper, data=data)
        
        elif isinstance(data, str):
            # create a specific condition given by a string
            return cls.from_str(grid, axis, upper=upper, condition=data)
        
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
        if self.homogeneous:
            if np.isscalar(self.value) and self.value == 0:
                return True
            
            elif rank == 0:
                if self.rank != 0:
                    raise RuntimeError('Expected scalar boundary condition but '
                                       f'got `{self.value}`')
                    
            elif np.isscalar(self.value):  # self.value != 0
                raise RuntimeError('Expected boundary condition of rank '
                                   f'{rank} but got scalar `{self.value}`')
                    
            elif self.value.shape != (self.grid.dim,) * rank:
                raise RuntimeError('Expected boundary condition of rank '
                                   f'{rank} but got shape `{self.value.shape}`')
                
        else:
            # inhomogeneous boundary conditions
            if rank != 0:
                raise NotImplementedError('Only homogeneous tensorial boundary '
                                          'conditions are supported')
                
        return True
            
            
    @abstractmethod
    def get_virtual_point(self, arr, idx: Tuple[int, ...] = None) -> float: pass
            
    @abstractmethod
    def get_virtual_point_evaluator(self) -> Callable: pass
    
    @abstractmethod
    def get_adjacent_evaluator(self) -> Callable: pass
            
    @property
    def differentiated(self) -> "BCBase": 
        """ BCBase: differentiated version of this boundary condition """
        raise NotImplementedError



class BCBase1stOrder(BCBase):
    """ represents a single boundary in an BoundaryPair instance """


    @abstractmethod
    def get_virtual_point_data(self) -> Tuple[Any, Any, int]: pass


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
                    data[1] * arr_1d[..., data[2]])
        else:
            return (data[0][bc_idx] +  # type: ignore
                    data[1][bc_idx] * arr_1d[..., data[2]])


    def get_virtual_point_evaluator(self) -> Callable:
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
        data = self.get_virtual_point_data()
        
        if self.homogeneous:
            @jit
            def virtual_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the virtual point at `idx` """
                arr_1d, _, _ = get_arr_1d(arr, idx)
                return (data[0] +  # type: ignore
                        data[1] * arr_1d[..., data[2]])
                
        else:
            @jit
            def virtual_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the virtual point at `idx` """
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                return (data[0][bc_idx] +  # type: ignore
                        data[1][bc_idx] * arr_1d[..., data[2]])
        
        return virtual_point  # type: ignore
    
    
    def get_adjacent_evaluator(self) -> Callable:
        """ returns a function evaluating the value adjacent to a given point 

        Returns:
            function: A function that takes the data array and an index marking
            the current point. The result is the data value at the adjacent 
            point along the axis associated with this boundary condition in the
            upper (lower) direction when `upper` is True (False).
        """
        size = self.grid.shape[self.axis]
        upper = self.upper
        dx = self.grid.discretization[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        
        if not isinstance(dx, numbers.Number):
            raise ValueError(f'Discretization along axis {self.axis} must be a '
                             f'number, not `{dx}`')

        # calculate necessary constants
        data_vp = self.get_virtual_point_data()
        
        if self.homogeneous:
            # the boundary condition does not depend on space
            if self.is_scalar:
                zero = 0.
            else:
                zero = np.zeros((self.grid.dim,) * self.rank)
            
            @jit
            def adjacent_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the point adjacent to `idx` """
                # extract the 1d array
                arr_1d, i, _ = get_arr_1d(arr, idx)

                # determine the parameters for evaluating adjacent point. Note
                # that this is an optimization that apparently accelerates the
                # computation. The alternative direct calculation (that is also
                # used in the inhomogeneous case below) turned out to be
                # significantly slower.
                if upper:
                    if i == size - 1:
                        data = data_vp
                    else:
                        data = (zero, 1., i + 1)
                else:
                    if i == 0:
                        data = data_vp
                    else:
                        data = (zero, 1., i - 1)
                
                # calculate the values
                return (data[0] +  # type: ignore
                        data[1] * arr_1d[..., data[2]])
                
        else:
            # the boundary condition is a function of space
            
            @jit
            def adjacent_point(arr, idx: Tuple[int, ...]) -> float:
                """ evaluate the point adjacent to `idx` """
                arr_1d, i, bc_idx = get_arr_1d(arr, idx)

                # determine the parameters for evaluating adjacent point
                if upper:
                    if i == size - 1:
                        val = (data_vp[0][bc_idx] +
                               data_vp[1][bc_idx] * arr_1d[..., data_vp[2]])
                    else:
                        val = arr_1d[..., i + 1]
                else:
                    if i == 0:
                        val = (data_vp[0][bc_idx] +
                               data_vp[1][bc_idx] * arr_1d[..., data_vp[2]])
                    else:
                        val = arr_1d[..., i - 1]
                
                return val  # type: ignore
        
        return adjacent_point  # type: ignore    
    


class DirichletBC(BCBase1stOrder):
    """ represents a boundary condition imposing the value """
    
    names = ['value', 'dirichlet']  # identifiers for this boundary condition

    
    def get_virtual_point_data(self) -> Tuple[Any, float, int]:
        """ return data suitable for calculating virtual points
        
        Returns:
            tuple: the data structure associated with this virtual point
        """        
        size = self.grid.shape[self.axis]
        
        value = 2 * self.value
        factor = -1 if np.isscalar(value) else -np.ones_like(self.value)
        if self.upper:
            idx = size - 1
        else:
            idx = 0
        return (value, factor, idx)
    
            
    @property
    def differentiated(self) -> BCBase:
        """ BCBase: differentiated version of this boundary condition """
        return NeumannBC(grid=self.grid, axis=self.axis, upper=self.upper,
                         value=np.zeros_like(self.value))



class NeumannBC(BCBase1stOrder):
    """ represents a boundary condition imposing the derivative in the outward
    normal direction of the boundary """
    
    names = ['derivative', 'neumann']  # identifiers for this boundary condition        

    
    def get_virtual_point_data(self) -> Tuple[Any, float, int]:
        """ return data suitable for calculating virtual points
            
        Returns:
            tuple: the data structure associated with this virtual point
        """        
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]
        
        value = dx * self.value
        factor = 1 if np.isscalar(value) else np.ones_like(self.value)
        if self.upper:
            idx = size - 1
        else:
            idx = 0
        return (value, factor, idx)


    @property
    def differentiated(self) -> BCBase:
        """ BCBase: differentiated version of this boundary condition """
        return CurvatureBC(grid=self.grid, axis=self.axis, upper=self.upper,
                           value=np.zeros_like(self.value))



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
    """
    
    names = ['mixed', 'robin']
    
    def __init__(self, grid: GridBase, axis: int, upper: bool, value=0,
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
        super().__init__(grid, axis, upper, value)
        # TODO: support spatially varying constant terms β
        self.const = float(const)  
        
    def __eq__(self, other):
        """ checks for equality neglecting the `upper` property """
        return super().__eq__(other) and self.const == other.const


    def copy(self, upper: Optional[bool] = None, value=None, const=None) \
            -> "MixedBC":
        """ return a copy of itself, but with a reference to the same grid """
        if upper is None:
            upper = self.upper
        if value is None:
            value = self._value_expression
        if const is None:
            const = self.const
        return self.__class__(grid=self.grid, axis=self.axis, upper=upper,
                              value=value, const=const)        
        
        
    def get_virtual_point_data(self) -> Tuple[Any, float, int]:
        """ return data suitable for calculating virtual points
            
        Returns:
            tuple: the data structure associated with this virtual point
        """        
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]
        
        factor = dx * self.value
        if np.isscalar(factor):
            if np.isinf(factor):
                value = 0
                factor = -1
            else:
                value = 2 * dx * self.const / (2 + factor)
                factor = (2 - factor) / (2 + factor)
        else:
            # calculate values assuming finite factor
            value = 2 * dx * self.const / (2 + factor)
            factor = (2 - factor) / (2 + factor)
            # correct at places of infinite values 
            value[np.isinf(factor)] = 0  # type: ignore
            factor[np.isinf(factor)] = -1
            
        if self.upper:
            idx = size - 1
        else:
            idx = 0
        return (value, factor, idx)
    


class BCBase2ndOrder(BCBase):
    """ abstract base class for boundary conditions of 2nd order """
    
    
    @abstractmethod
    def get_virtual_point_data(self) -> Tuple[Any, Any, int, Any, int]:
        """ return data suitable for calculating virtual points
        
        Returns:
            tuple: the data associated with this virtual point 
        """ 

    
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
    
        
    def get_virtual_point_evaluator(self) -> Callable:
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
            @jit
            def virtual_point(arr, idx: Tuple[int, ...]):
                """ evaluate the virtual point at `idx` """
                arr_1d, _, _ = get_arr_1d(arr, idx)
                
                return (data[0] +
                        data[1] * arr_1d[..., data[2]] +
                        data[3] * arr_1d[..., data[4]])
            
        else:
            @jit
            def virtual_point(arr, idx: Tuple[int, ...]):
                """ evaluate the virtual point at `idx` """
                arr_1d, _, bc_idx = get_arr_1d(arr, idx)
                
                return (data[0][bc_idx] +
                        data[1][bc_idx] * arr_1d[..., data[2]] +
                        data[3][bc_idx] * arr_1d[..., data[4]])
            
        return virtual_point  # type: ignore
        

    def get_adjacent_evaluator(self) -> Callable:
        """ returns a function evaluating the value adjacent to a given point 

        Returns:
            function: A function that takes the data array and an index marking
            the current point. The result is the data value at the adjacent 
            point along the axis associated with this boundary condition in the
            upper (lower) direction when `upper` is True (False).
        """
        size = self.grid.shape[self.axis]
        upper = self.upper
        dx = self.grid.discretization[self.axis]
        get_arr_1d = _make_get_arr_1d(self.grid.num_axes, self.axis)
        
        if size < 2:
            raise ValueError('Need at least two support points along axis '
                             f'{self.axis} to apply boundary conditions')
        if not isinstance(dx, numbers.Number):
            raise ValueError(f'Discretization along axis {self.axis} must be a '
                             f'number, not `{dx}`')

        # calculate necessary constants
        data_vp = self.get_virtual_point_data()
        
        if self.homogeneous:
            # the boundary condition does not depend on space
            if self.is_scalar:
                zero = 0.
            else:
                zero = np.zeros((self.grid.dim,) * self.rank)
            
            @jit
            def adjacent_point(arr, idx: Tuple[int, ...]):
                """ evaluate the point adjacent to `idx` """
                # extract the 1d array
                arr_1d, i, _ = get_arr_1d(arr, idx)

                # determine the parameters for evaluating adjacent point
                if upper:
                    if i == size - 1:
                        data = data_vp
                    else:
                        data = (zero, 1., i + 1, 0., 0)
                else:
                    if i == 0:
                        data = data_vp
                    else:
                        data = (zero, 1., i - 1, 0., 0)
                
                # calculate the values
                return (data[0] +
                        data[1] * arr_1d[..., data[2]] +
                        data[3] * arr_1d[..., data[4]])
                
        else:
            # the boundary condition is a function of space
            
            @jit
            def adjacent_point(arr, idx: Tuple[int, ...]):
                """ evaluate the point adjacent to `idx` """
                arr_1d, i, bc_idx = get_arr_1d(arr, idx)

                # determine the parameters for evaluating adjacent point
                if upper:
                    if i == size - 1:
                        val = (data_vp[0][bc_idx] +
                               data_vp[1][bc_idx] * arr_1d[..., data_vp[2]] + 
                               data_vp[3][bc_idx] * arr_1d[..., data_vp[4]])
                    else:
                        val = arr_1d[..., i + 1]
                else:
                    if i == 0:
                        val = (data_vp[0][bc_idx] +
                               data_vp[1][bc_idx] * arr_1d[..., data_vp[2]] +
                               data_vp[3][bc_idx] * arr_1d[..., data_vp[4]])
                    else:
                        val = arr_1d[..., i - 1]
                
                return val
        
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
