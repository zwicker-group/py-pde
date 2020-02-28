'''
Defines base classes


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import functools
import operator
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import (Tuple, Callable, Optional, Union, Any, Dict,
                    List, TypeVar, Iterator, TYPE_CHECKING)  # @UnusedImport

import numpy as np
from scipy import interpolate, ndimage

from ..grids import CylindricalGrid, SphericalGrid
from ..grids.base import GridBase, discretize_interval
from ..grids.cartesian import CartesianGridBase
from ..tools.numba import jit
from ..tools.cache import cached_method


if TYPE_CHECKING:
    from .scalar import ScalarField  # @UnusedImport


ArrayLike = Union[np.ndarray, float]
OptionalArrayLike = Optional[ArrayLike]
T1 = TypeVar('T1', bound='FieldBase')



class FieldBase(metaclass=ABCMeta):
    """ abstract base class for describing (discretized) fields
    
    Attributes:
        grid (:class:`~pde.grids.GridBase`):
            The underlying grid defining the discretization
        data (:class:`numpy.ndarray`):
            Data values at the support points of the grid
        label (str):
            Name of the field
    """ 
    
    _subclasses: Dict[str, Any] = {}  # all classes inheriting from this
    readonly = False
    
    
    def __init__(self, grid: GridBase, data: OptionalArrayLike = None,
                 label: Optional[str] = None):
        """ 
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            data (array, optional):
                Field values at the support points of the grid
            label (str, optional):
                Name of the field
        """
        self.grid = grid
        self._data: Any = data
        self.label = label
        self._logger = logging.getLogger(self.__class__.__module__)


    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls


    @classmethod
    def from_state(cls, state: Union[str, Dict[str, Any]],
                   grid: GridBase, data=None) -> "FieldBase":
        """ create a field from given state.
        
        Args:
            state (str or dict): State from which the instance is created. If 
                `state` is a string, it is decoded as JSON.
            grid (:class:`~pde.grids.GridBase`):
                The grid that is used to describe the field
            data (:class:`numpy.ndarray`, optional): Data values at the support
                points of the grid that define the field.
        """
        # decode the json data
        if isinstance(state, str):
            import json
            state = dict(json.loads(state))
        
        # create the instance of the correct class
        class_name = state.pop('class')
        if class_name == cls.__name__:
            raise RuntimeError('Cannot reconstruct abstract class' 
                               f'`{class_name}`')
        field_cls = cls._subclasses[class_name]
        return field_cls.from_state(state, grid, data=data)  # type: ignore

    
    @classmethod
    def from_file(cls, filename: str) -> "FieldBase":
        """ create field by reading file
        
        Args:
            filename (str): Path to the file being read
        """
        import h5py
        from .collection import FieldCollection
        
        with h5py.File(filename, "r") as fp:
            if 'class' in fp.attrs:
                # this should be a field collection
                assert fp.attrs['class'] == 'FieldCollection'
                obj = FieldCollection._from_dataset(fp)
                
            elif len(fp) == 1:
                # a single field is stored in the data
                dataset = fp[list(fp.keys())[0]]  # retrieve only dataset
                obj = cls._from_dataset(dataset)  # type: ignore
                
            else:
                raise RuntimeError('Multiple data fields were found in the '
                                   'file but no FieldCollection is expected')
        return obj
                
                
    @classmethod
    def _from_dataset(cls, dataset) -> "FieldBase":
        """ construct a field by reading data from an hdf5 dataset """
        class_name = dataset.attrs['class']
        field_cls = cls._subclasses[class_name]
        
        grid = GridBase.from_state(dataset.attrs['grid'])
        label = dataset.attrs['label'] if 'label' in dataset.attrs else None
        return field_cls(grid, data=dataset, label=label)  # type: ignore


    def to_file(self, filename: str):
        """ store field in hdf5 file
        
        Args:
            filename (str): Path where the data is stored
        """
        import h5py
        with h5py.File(filename, "w") as fp:
            self._write_hdf_dataset(fp)


    def _write_hdf_dataset(self, fp, key: str = 'data'):
        """ write data to a given hdf5 file pointer `fp` """
        dataset = fp.create_dataset(key, data=self.data)
        dataset.attrs['class'] = self.__class__.__name__
        if self.label:      
            dataset.attrs['label'] = str(self.label)      
        dataset.attrs['grid'] = self.grid.state_serialized      

    
    @abstractmethod
    def copy(self: T1, data=None, label: str = None) -> T1: pass
            
         
    def assert_field_compatible(self, other: 'FieldBase',
                                accept_scalar: bool = False):
        """ checks whether `other` is compatible with the current field
        
        Args:
            other (FieldBase): Other field this is compared to
            accept_scalar (bool, optional): Determines whether it is acceptable
                that `other` is an instance of
                :class:`~pde.fields.ScalarField`.
        """
        from .scalar import ScalarField  # @Reimport
        
        # check whether they are the same class
        class_compatible = (self.__class__ == other.__class__ or
                            (accept_scalar and isinstance(other, ScalarField)))
        if not class_compatible:
            raise TypeError('Fields are incompatible')
        
        # check whether the associated grids are identical
        if not self.grid.compatible_with(other.grid):
            raise ValueError('Grids incompatible')
    
         
    @property
    def data(self):
        """ :class:`numpy.ndarray`: discretized data at the support points """
        return self._data
    
    @data.setter
    def data(self, value):
        if self.readonly:
            raise RuntimeError(f'Cannot write to {self.__class__.__name__}')
        if isinstance(value, FieldBase):
            # copy data into current field
            self.assert_field_compatible(value, accept_scalar=True)
            self._data[:] = value.data
        else:
            self._data[:] = value
        
        
    @property
    def state(self) -> dict:
        """ dict: current state of this instance """
        return {'label': self.label}
    
        
    @property
    def state_serialized(self) -> str:
        """ str: a json serialized version of the field """
        import json
        state = self.state
        state['class'] = self.__class__.__name__
        return json.dumps(state)
    

    @property
    def _data_flat(self):
        """ :class:`numpy.ndarray`: flat version of discretized data """
        # flatten the first dimension of the internal data
        return self._data.reshape(-1, *self.grid.shape)
    
    @_data_flat.setter
    def _data_flat(self, value):
        """ set the data from a value from a collection """
        if self.readonly:
            raise RuntimeError(f'Cannot write to {self.__class__.__name__}')
        # simply set the data -> this might need to be overwritten
        self._data = value
            
    
    def __eq__(self, other):
        """ test for equality """
        return (self.state == other.state and
                np.array_equal(self.data, other.data))
    

    def __neg__(self):
        """ return the negative of the current field """
        return self.copy(data=-self.data)


    def _binary_operation(self, other, op: Callable,
                          scalar_second: bool = True) -> "FieldBase":
        """ perform a binary operation between this field and `other`
        
        Args:
            other (number of FieldBase):
                The second term of the operator
            op (callable):
                A binary function calculating the result
            scalar_second (bool):
                Flag determining whether the second operator must be a scalar
                
        Returns:
            FieldBase: An field that contains the result of the operation. If
            `scalar_second == True`, the type of FieldBase is the same as `self`
        """
        if isinstance(other, FieldBase):
            # right operator is a field
            from .scalar import ScalarField  # @Reimport
            
            if scalar_second:
                # right operator must be a scalar 
                if not isinstance(other, ScalarField):
                    raise TypeError('Right operator must be a scalar field')
                self.grid.assert_grid_compatible(other.grid)
                result: FieldBase = self.copy(op(self.data, other.data))
                
            elif isinstance(self, ScalarField):
                # left operator is a scalar field (right can be tensor)
                self.grid.assert_grid_compatible(other.grid)
                result = other.copy(op(self.data, other.data))
                
            else:
                # left operator is tensor and right one might be anything
                self.assert_field_compatible(other, accept_scalar=True)
                result = self.copy(op(self.data, other.data))
                
        else:
            # the second operator is a number or a numpy array
            result = self.copy(op(self.data, other))
        return result        

    
    def _binary_operation_inplace(self, other, op_inplace: Callable,
                                  scalar_second: bool = True) -> "FieldBase":
        """ perform an in-place binary operation between this field and `other`
        
        Args:
            other (number of FieldBase):
                The second term of the operator
            op_inplace (callable):
                A binary function storing its result in the first argument
            scalar_second (bool):
                Flag determining whether the second operator must be a scalar.
                
        Returns:
            FieldBase: The field `self` with updated data
        """
        if self.readonly:
            raise RuntimeError(f'Cannot write to {self.__class__.__name__}')
        
        if isinstance(other, FieldBase):
            # right operator is a field
            from .scalar import ScalarField  # @Reimport
            
            if scalar_second:
                # right operator must be a scalar 
                if not isinstance(other, ScalarField):
                    raise TypeError('Right operator must be a scalar field')
                self.grid.assert_grid_compatible(other.grid)
            else:
                # left operator is tensor and right one might be anything
                self.assert_field_compatible(other, accept_scalar=True)
                
            op_inplace(self.data, other.data)
                
        else:
            # the second operator is a number or a numpy array
            op_inplace(self.data, other)
            
        return self        

    
    def __add__(self, other) -> "FieldBase":
        """ add two fields """
        return self._binary_operation(other, operator.add, scalar_second=False)
    __radd__ = __add__


    def __iadd__(self, other) -> "FieldBase":
        """ add `other` to the current field """
        return self._binary_operation_inplace(other, operator.iadd,
                                              scalar_second=False)
    
    
    def __sub__(self, other) -> "FieldBase":
        """ subtract two fields """
        return self._binary_operation(other, operator.sub, scalar_second=False)

    
    def __rsub__(self, other) -> "FieldBase":
        """ subtract two fields """
        return self._binary_operation(other, lambda x, y: y - x,
                                      scalar_second=False)


    def __isub__(self, other) -> "FieldBase":
        """ add `other` to the current field """
        return self._binary_operation_inplace(other, operator.isub,
                                              scalar_second=False)
    
    
    def __mul__(self, other) -> "FieldBase":
        """ multiply field by value """
        return self._binary_operation(other, operator.mul, scalar_second=False)
    __rmul__ = __mul__
    
       
    def __imul__(self, other) -> "FieldBase":
        """ multiply field by value """
        return self._binary_operation_inplace(other, operator.imul,
                                              scalar_second=False)

    
    def __truediv__(self, other) -> "FieldBase":
        """ divide field by value """
        return self._binary_operation(other, operator.truediv,
                                      scalar_second=True)
    
    
    def __itruediv__(self, other) -> "FieldBase":
        """ divide field by value """
        return self._binary_operation_inplace(other, operator.itruediv,
                                              scalar_second=True)
       
       
    def __pow__(self, exponent: float) -> "FieldBase":
        """ raise data of the field to a certain power """
        if not np.isscalar(exponent):
            raise NotImplementedError('Only scalar exponents are supported')
        return self.copy(data=self.data ** exponent)
    
    
    def __ipow__(self, exponent: float) -> "FieldBase":
        """ raise data of the field to a certain power in-place """
        if self.readonly:
            raise RuntimeError(f'Cannot write to {self.__class__.__name__}')
        
        if not np.isscalar(exponent):
            raise NotImplementedError('Only scalar exponents are supported')
        self.data **= exponent
        return self


    def get_line_data(self, extract: str = 'auto') -> Dict[str, Any]:
        """ return data for a line plot of the field
        
        Args:
            extract (str):
                The method used for extracting the line data. See the docstring
                of the grid method `get_line_data` to find supported values.
        
        Returns:
            dict: Information useful for performing a line plot of the field
        """
        result = self.grid.get_line_data(self.data, extract=extract)
        if 'label_y' in result and result['label_y']:
            if self.label:
                result['label_y'] = f"{self.label} ({result['label_y']})"
        else:
            result['label_y'] = self.label
        return result  # type: ignore  
        
        
    def get_image_data(self, **kwargs) -> Dict[str, Any]:
        r""" return data for plotting an image of the field
        
        Args:
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`
        
        Returns:
            dict: Information useful for plotting an image of the field        
        """
        result = self.grid.get_image_data(self.data, **kwargs)  # type: ignore 
        result['title'] = self.label
        return result  # type: ignore  
    
    
    def get_vector_data(self, **kwargs) -> Dict[str, Any]:
        r""" return data for a vector plot of the field
        
        Args:
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`
        
        Returns:
            dict: Information useful for plotting an vector field        
        """
        raise NotImplementedError()
    
    
    def plot_line(self, **kwargs):
        r""" visualize a field using a 1d cut """
        raise NotImplementedError()
    
    
    def plot_image(self, ax=None,
                   colorbar: bool = False,
                   transpose: bool = False,
                   title: Optional[str] = None, **kwargs):
        r""" visualize an 2d image of the field

        Args:
            ax: Figure axes to be used for plotting. If `None`, a new figure is
                created
            colorbar (bool): determines whether a colorbar is shown
            transpose (bool): determines whether the transpose of the data
                should be plotted.
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            \**kwargs: Additional keyword arguments are passed to
                `matplotlib.pyplot.imshow`.
                
        Returns:
            Result of `plt.imshow`
        """
        raise NotImplementedError()
    
    
    def plot_vector(self, method: str, ax=None, transpose: bool = False,
                    title: Optional[str] = None, **kwargs):
        r""" visualize a 2d vector field

        Args:
            method (str): Plot type that is used. This can be either `quiver`
                or `streamplot`.
            ax: Figure axes to be used for plotting. If `None`, a new figure is
                created
            transpose (bool): determines whether the transpose of the data
                should be plotted.
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            \**kwargs: Additional keyword arguments are passed to
                `matplotlib.pyplot.quiver` or `matplotlib.pyplot.streamplot`.
                
        Returns:
            Result of `plt.quiver` or `plt.streamplot`
        """
        raise NotImplementedError()
    
            
    def plot(self, kind: str = 'auto', filename=None, **kwargs):
        r""" visualize the field
        
        Args:
            kind (str): Determines the visualizations. Supported values are
                `image`,  `line`, or `vector`. Alternatively, `auto` determines
                the best visualization based on the field itself.
            filename (str, optional): If given, the plot is written to the
                specified file. Otherwise, the plot might show directly in an
                interactive matplotlib session or `matplotlib.pyplot.show()`
                might be used to display the graphics.
            ax: Figure axes to be used for plotting. If `None`, a new figure is
                created
            \**kwargs: All additional keyword arguments are forwarded to the
                actual plotting functions.
                
        Returns:
            The result of the respective matplotlib plotting function
        """
        if kind == 'auto':
            # determine best plot for this field
            if (isinstance(self, DataFieldBase) and self.rank == 1 and
                    self.grid.dim == 2):
                kind = 'vector'
            elif len(self.grid.shape) == 1:
                kind = 'line'
            else:
                kind = 'image'

        # do the actual plotting
        if kind == 'image':
            res = self.plot_image(**kwargs)
        elif kind == 'line':
            res = self.plot_line(**kwargs)
        elif kind == 'vector':
            res = self.plot_vector(**kwargs)
        else:
            raise ValueError(f'Unsupported plot `{kind}`. Possible choices are '
                             '`image`, `line`, `vector`, or `auto`.')
            
        # store the result to a file if requested
        if filename:
            import matplotlib.pyplot as plt
            plt.savefig(filename)
            
        return res
                    


T2 = TypeVar('T2', bound='DataFieldBase')


class DataFieldBase(FieldBase, metaclass=ABCMeta):
    """ abstract base class for describing fields of single entities
    
    Attributes:
        grid (:class:`~pde.grids.GridBase`):
            The underlying grid defining the discretization
        data (:class:`numpy.ndarray`):
            Data values at the support points of the grid
        shape (tuple):
            Shape of the `data` field
        label (str):
            Name of the field
    """ 
         
    rank: int  # the rank of the tensor field
    _allocate_memory = True  # determines whether the instances allocated memory
             

    def __init__(self, grid: GridBase,
                 data: OptionalArrayLike = None,
                 label: Optional[str] = None):
        """ 
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            data (array, optional):
                Field values at the support points of the grid
            label (str, optional):
                Name of the field
        """
        # determine data shape
        shape = (grid.dim,) * self.rank + grid.shape
        
        if self._allocate_memory:
            # class manages its own data, which therefore needs to be allocated
            if data is None:
                data = np.zeros(shape, dtype=np.double)
            elif isinstance(data, DataFieldBase):
                # we need to make a copy to make sure the data is writeable 
                data = np.array(np.broadcast_to(data.data, shape),
                                dtype=np.double, copy=True)
            else:
                # we need to make a copy to make sure the data is writeable 
                data = np.array(np.broadcast_to(data, shape), dtype=np.double,
                                copy=True)
                
        elif data is not None:
            # class does not manage its own data
            raise ValueError(f"{self.__class__.__name__} does not support data "
                              "assignment.")
            
        super().__init__(grid, data=data, label=label)
            
                     
    def __repr__(self):
        """ return instance as string """
        class_name = self.__class__.__name__
        result = f'{class_name}(grid={self.grid!r}, data={self.data}'
        if self.label:
            result += f', label="{self.label}"'
        return result + ')'

    
    def __str__(self):
        """ return instance as string """
        result = f'{self.__class__.__name__}(grid={self.grid}, ' \
                 f'data=Array{self.data.shape}'
        if self.label:
            result += f', label="{self.label}"'
        return result + ')'


    @classmethod
    def random_uniform(cls, grid: GridBase, vmin: float = 0, vmax: float = 1,
                       label: Optional[str] = None, seed: Optional[int] = None):
        """ create field with uniform distributed random values
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            vmin (float): Smallest random value
            vmax (float): Largest random value
            label (str, optional): Name of the field
            seed (int, optional): Seed of the random number generator. If
                `None`, the current state is not changed.
        """
        shape = (grid.dim,) * cls.rank + grid.shape
        if seed is not None:
            np.random.seed(seed)
        data = np.random.uniform(vmin, vmax, shape)
        return cls(grid, data, label=label)
    
    
    @classmethod
    def random_normal(cls, grid: GridBase, mean: float = 0, std: float = 1,
                      scaling: str = 'physical', label: Optional[str] = None,
                      seed: Optional[int] = None):
        """ create field with normal distributed random values
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            mean (float): Mean of the Gaussian distribution
            std (float): Standard deviation of the Gaussian distribution
            scaling (str): Determines how the noise is scaled. Possible values
                are 'none' (values are drawn from a normal distribution with
                given mean and standard deviation) or 'physical' (the variance
                of the random number is scaled by the inverse volume of the grid
                cell; this is useful for physical quantities, which vary less in
                larger volumes).            
            label (str, optional): Name of the field
            seed (int, optional): Seed of the random number generator. If
                `None`, the current state is not changed.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if scaling == 'none':
            noise_scale = std
        elif scaling == 'physical':
            noise_scale = std / np.sqrt(grid.cell_volume_data)
        else:
            raise ValueError(f'Unknown noise scaling {scaling}')
            
        shape = (grid.dim,) * cls.rank + grid.shape
        data = mean + noise_scale * np.random.randn(*shape)
        return cls(grid, data, label=label)

    
    @classmethod
    def random_harmonic(cls, grid: GridBase,
                        modes: int = 3,
                        harmonic=np.cos,
                        axis_combination=np.multiply,
                        label: Optional[str] = None,
                        seed: Optional[int] = None):
        r""" create a random field build from harmonics
        
        Such fields can be helpful for testing differential operators. They
        serve as random input without the high frequencies that come with
        random uncorrelated fields.
        
        With the default settings, the resulting field :math:`c_i(\mathbf{x})`
        is given by
        
        .. math::
            c_i(\mathbf{x}) = \prod_{\alpha=1}^N \sum_{j=1}^M a_{ij\alpha}
                \cos\left(\frac{2 \pi x_\alpha}{j L_\alpha}\right) \;,
            
        where :math:`N` is the number of spatial dimensions, each with length 
        :math:`L_\alpha`, :math:`M` is the number of modes given by `modes`, and
        :math:`a_{ij\alpha}` are random amplitudes, chosen from a uniform
        distribution over the interval [0, 1].
        
        Note that the product could be replaced by a sum when
        `axis_combination = numpy.add` and the :math:`\cos()` could be any other
        function given by the parameter `harmonic`.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            modes (int): Number :math:`M` of harmonic modes
            harmonic (callable): Determines which harmonic function is used.
                Typical values are `numpy.sin` and `numpy.cos`, which basically
                relate to different boundary conditions applied at the grid
                boundaries.
            axis_combination (callable): Determines how values from different
                axis are combined. Typical choices are `numpy.multiply` and
                `numpy.add` resulting in products and sums of the values along
                axes, respectively.
            label (str, optional): Name of the field
            seed (int, optional): Seed of the random number generator. If
                `None`, the current state is not changed.
        """
        tensor_shape = (grid.dim,) * cls.rank
        if seed is not None:
            np.random.seed(seed)
    
        data = np.empty(tensor_shape + grid.shape)
        # determine random field for each component
        for index in np.ndindex(*tensor_shape):
            data_axis = []
            # random harmonic function along each axis
            for i in range(len(grid.axes)):
                # choose wave vectors
                ampl = np.random.random(modes)  # amplitudes
                x = discretize_interval(0, 2*np.pi, grid.shape[i])[0]
                data_axis.append(sum(a * harmonic(n * x)
                                     for n, a in enumerate(ampl, 1)))
            # full dataset is product of values along axes
            data[index] = functools.reduce(axis_combination.outer, data_axis)
            
        return cls(grid, data, label=label)


    @classmethod
    def random_colored(cls, grid: GridBase,
                       exponent: float = 0,
                       scale: float = 1,
                       label: Optional[str] = None, 
                       seed: Optional[int] = None):
        r""" create a field of random values that obey
        
        .. math::
            \langle c_i(\boldsymbol k) c_j(\boldsymbol k’) \rangle =
                \Gamma^2 |\boldsymbol k|^\nu \delta_{ij}
                \delta(\boldsymbol k - \boldsymbol k’)
                
        in spectral space. The special case :math:`\nu = 0` corresponds to white
        noise. Note that the components of vector or tensor fields are
        uncorrelated.
         
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            exponent (float):
                Exponent :math:`\nu` of the power spectrum
            scale (float):
                Scaling factor :math:`\Gamma` determining noise strength
            label (str, optional): Name of the field
            seed (int, optional): Seed of the random number generator. If
                `None`, the current state is not changed.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # create function making colored noise    
        from pde.tools.spectral import make_colored_noise
        make_noise = make_colored_noise(grid.shape, dx=grid.discretization,
                                        exponent=exponent, scale=scale)
        
        # create random fields for each tensor component
        tensor_shape = (grid.dim,) * cls.rank
        data = np.empty(tensor_shape + grid.shape)
        # determine random field for each component
        for index in np.ndindex(*tensor_shape):
            data[index] = make_noise()
         
        return cls(grid, data, label=label)
    
        
    @classmethod
    def from_state(cls, state: Dict[str, Any], grid: GridBase,  # type: ignore
                   data=None) -> "DataFieldBase":
        """ create a field from given state.
        
        Args:
            state (str or dict): State from which the instance is created. If 
                `state` is a string, it is decoded as JSON.
            grid (:class:`~pde.grids.GridBase`):
                The grid that is used to describe the field
            data (:class:`numpy.ndarray`, optional): Data values at the support
                points of the grid that define the field.
        """
        return cls(grid, data=data, **state)
            
        
    def copy(self: T2, data=None, label: str = None) -> T2:
        """ return a copy of the data, but not of the grid
        
        Args:
            data (:class:`numpy.ndarray`, optional): Data values at the support
                points of the grid that define the field.
            label (str, optional): Name of the copied field
        """
        if label is None:
            label = self.label
        if data is None:
            data = self.data
        # the actual data will be copied in our __init__ method
        return self.__class__(self.grid, data=data, label=label)
    
    
    @property
    def data_shape(self) -> Tuple[int, ...]:
        """ tuple: the shape of the data at each grid point """
        return (self.grid.dim,) * self.rank
    

    def _make_interpolator_scipy(self, **kwargs) -> Callable:
        r""" returns a function that can be used to interpolate values.
        
        This uses scipy.interpolate.RegularGridInterpolator and the
        interpolation method can thus be chosen when calling the returned
        function using the `method` keyword argument. Keyword arguments are
        directly forwarded to the constructor of `RegularGridInterpolator`.
        
        Note that this interpolator does not respect periodic boundary
        conditions, yet.
        
        Args:
            \**kwargs: All keyword arguments are forwarded to
                :class:`scipy.interpolate.RegularGridInterpolator`
                
        Returns:
            A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. 
        """
        coords_src = self.grid.axes_coords
        grid_dim = len(self.grid.axes)
        
        if self.rank == 0:
            # scalar field => data layout is already usable
            data = self.data
            revert_shape = False
        else:
            # spatial dimensions need to come first => move data to last axis
            assert self.data.shape[:-grid_dim] == self.data_shape
            data_flat = self._data_flat
            data_flat = np.moveaxis(data_flat, 0, -1)
            new_shape = self.grid.shape + (-1,)
            data = data_flat.reshape(new_shape)
            assert data.shape[-1] == self.grid.dim ** self.rank
            revert_shape = True
            
        # prepare the interpolator
        intp = interpolate.RegularGridInterpolator(coords_src, data, **kwargs)
        
        # determine under which conditions the axes can be squeezed
        if grid_dim == 1:
            scalar_dim = 0
        else:
            scalar_dim = 1
            
        # introduce wrapper function to process arrays
        def interpolator(point, **kwargs):
            """ return the interpolated value at the position `point` """
            point = np.atleast_1d(point)
            # apply periodic boundary conditions to grid point
            point = self.grid.normalize_point(point, reduced_coords=True)
            out = intp(point, **kwargs)
            if point.ndim == scalar_dim or point.ndim == point.size == 1:
                out = out[0]
            if revert_shape:
                # revert the shuffling of spatial and local axes
                out = np.moveaxis(out, point.ndim - 1, 0) 
                out = out.reshape(self.data_shape + point.shape[:-1])
            
            return out
            
        return interpolator


    def _make_interpolator_compiled(self, method: str = 'linear',
                                    bc='natural') -> Callable:
        """ return a compiled interpolator
        
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
            arbitrary positions within the space of the grid. 
        """
        if method != 'linear':
            raise ValueError(f'Interpolation method `{method}` is not '
                             'supported. `linear` is the only possible choice.')
        
        grid = self.grid
        grid_dim = len(grid.axes)
        data_shape = self.data_shape

        # create an interpolator for a single point
        interpolate_single = grid.make_interpolator_compiled(method, bc)
        
        # define wrapper function to always access current data of field
        @jit
        def interpolate_many(data, point):
            """ return the interpolated value at the position `point`
            
            Args:
                point (:class:`numpy.ndarray`): The list of points. This
                    function only accepts 2d lists of points
            """
            out = np.empty(data_shape + (point.shape[0],))
            for i in range(point.shape[0]):
                out[..., i] = interpolate_single(data, point[i, :])
            return out
        

        def interpolator(point):
            """ return the interpolated value at the position `point` """
            # turn points into a linear array
            point = np.atleast_1d(point)
            if point.shape[-1] != grid_dim:
                raise ValueError('Invalid points for interpolation.'
                                 '(Dimension mismatch)')
            points_shape = point.shape[:-1]  # keep original shape of points
                
            num_points = functools.reduce(operator.mul, points_shape, 1)
            point = point.reshape(num_points, grid_dim)
            
            out = interpolate_many(self.data, point)
            return out.reshape(self.data_shape + points_shape)
        
        return interpolator


    @cached_method()
    def make_interpolator(self, method: str = 'numba_linear', **kwargs):
        r""" returns a function that can be used to interpolate values.
        
        Args:
            method (str): Determines the method being used for interpolation.
                Possible values are:
                
                * `scipy_nearest`: Use scipy to interpolate to nearest neighbors
                * `scipy_linear`: Linear interpolation using scipy
                * `numba_linear`: Linear interpolation using numba (default)
                  
            \**kwargs: Additional keyword arguments are passed to the individual
                interpolator methods and can be used to further affect the
                behavior.
                  
        The scipy implementations use scipy.interpolate.RegularGridInterpolator
        and thus do not respect boundary conditions. Additional keyword
        arguments are directly forwarded to the constructor of
        `RegularGridInterpolator`.
        
        The numba implementation respect boundary conditions, which can be set
        using the `bc` keywords argument. Supported values are the same as for
        the operators, e.g., the Laplacian. If no boundary conditions are
        specified,  natural boundary conditions are assumed, which are periodic
        conditions for periodic axes and Neumann conditions otherwise.
        
        Returns:
            A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. 
        """
        if method.startswith('scipy'):
            return self._make_interpolator_scipy(method=method[6:], **kwargs)
        elif method.startswith('numba'):
            return self._make_interpolator_compiled(method=method[6:], **kwargs)
        else:
            raise ValueError(f'Unknown interpolation method `{method}`')
            

    def interpolate(self, point, **kwargs):
        r""" interpolate the field to points between support points
        
        Args:
            point (:class:`numpy.ndarray`):
                The points at which the values should be obtained. This is given
                in grid coordinates.
            \**kwargs:
                Additional keyword arguments are forwarded to the method
                :meth:`DataFieldBase.make_interpolator`.
                
        Returns:
            :class:`numpy.ndarray`: the values of the field
        """
        return self.make_interpolator(**kwargs)(point)


    def interpolate_to_grid(self: T2, grid: GridBase,
                            normalized: bool = False,
                            method: str = 'linear',
                            label: Optional[str] = None) -> T2:
        """ interpolate the data of this field to another grid.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            normalized (bool): Specifies whether the interpolation uses the true
                grid positions, potentially filling in undefined locations with
                zeros. Alternatively, if `normalized = True`, the data is
                stretched so both fields cover the same area.
            method (str): Specifies interpolation method, e.g. 'linear' or
                'nearest' 
            label (str, optional): Name of the returned field
            
        Returns:
            Field of the same rank as the current one.
        """
        if self.grid.dim != grid.dim:
            raise ValueError(f'Grid dimensions are incompatible '
                             f'({self.grid.dim:d} != {grid.dim:d})')
        # check if both grids are Cartesian
        src_cart = isinstance(self.grid, CartesianGridBase)
        dst_cart = isinstance(grid, CartesianGridBase)
            
        if isinstance(self.grid, (CylindricalGrid, SphericalGrid)) and dst_cart:
            # special interpolation defined by the grid class
            if normalized:
                self._logger.warning('Normalized interpolation is not '
                                     'supported for Curvilinear to Cartesian '
                                     'mapping')
            if method != 'linear':
                self._logger.warning('Setting interpolation method is only '
                                     'supported for Cartesian grids')
            
            def interpolator(data_slice):
                """ interpolates scalar data """
                return self.grid.interpolate_to_cartesian(data_slice, grid)
        
        elif (src_cart and dst_cart) or self.grid.__class__ == grid.__class__:
            # grid data is directly compatible
            
            # determine the coordinates of source and destination
            if normalized:
                coords_src = tuple(np.linspace(0, 1, len(c))
                                   for c in self.grid.axes_coords)
                coords_dst = [np.linspace(0, 1, s) for s in grid.shape]
                coords_dst = np.meshgrid(*coords_dst, indexing='ij')
                coords_dst = np.moveaxis(coords_dst, 0, -1)
            else:
                coords_src = self.grid.axes_coords
                coords_dst = grid.cell_coords
                
            def interpolator(data_slice):
                """ interpolates scalar data """
                return interpolate.interpn(
                    coords_src, data_slice, coords_dst, method=method,
                    bounds_error=False, fill_value=0)

        else: 
            # this type of interpolation is not supported
            raise NotImplementedError('Grid types are incompatible')
    
        # interpolate the actual data
        data_src = self._data_flat
        data_dst = np.empty((len(data_src), ) + grid.shape)
        for i in range(len(data_src)):
            data_dst[i] = interpolator(data_src[i])
    
        result = self.__class__(grid, label=label)
        result._data_flat = data_dst
        return result
    
    
    def add_interpolated(self, point, amount):
        """ adds an (integrated) value to the field at an interpolated position
        
        Args:
            point (:class:`numpy.ndarray`):
                The point inside the grid where the value is added. This is
                given in grid coordinates.
            amount (float or :class:`numpy.ndarray`):
                The amount that will be added to the field. The value describes
                an integrated quantity (given by the field value times the
                discretization volume). This is important for consistency with
                different discretizations and in particular grids with
                non-uniform discretizations.
        """
        point = np.atleast_1d(point)
        amount = np.broadcast_to(amount, self.data_shape)
        grid = self.grid
        grid_dim = len(grid.axes)
    
        if point.size != grid_dim or point.ndim != 1:
            raise ValueError(f'Dimension mismatch for point {point}')
        
        point = grid.normalize_point(point, reduced_coords=True)
    
        low = np.array(grid.axes_bounds)[:, 0]
        c_l, d_l = np.divmod((point - low) / grid.discretization - 0.5, 1.)
        c_l = c_l.astype(np.int)
        w_l = 1 - d_l  # weights of the low point
        w_h = d_l      # weights of the high point
    
        # determine the total weight in first iteration
        total_weight = 0
        cells = []
        for i in np.ndindex(*((2,) * grid_dim)):
            coords = np.choose(i, [c_l, c_l + 1])
            if np.all(coords >= 0) and np.all(coords < grid.shape):
                weight = np.prod(np.choose(i, [w_l, w_h]))
                total_weight += weight
                cells.append((tuple(coords), weight))
                
        if total_weight == 0:
            raise ValueError('Point lies outside grid')
    
        # alter each point in second iteration
        for coords, weight in cells:
            chng = weight * amount / (total_weight * grid.cell_volumes[coords])
            self.data[(Ellipsis,) + coords] += chng                                                  
        
        
    def apply(self: T2, func: Callable, out: Optional[T2] = None,
              label: str = None) -> T2:
        """ applies a function to the data and returns it as a field
        
        Args:
            func (callable): The (vectorized) function being applied to the
                data.
            out (FieldBase, optional): Optional field into which the data is
                written
            label (str, optional): Name of the returned field

        Returns:
            Field with new data. This is stored at `out` if given. 
        """
        if out is None:
            return self.copy(data=func(self.data), label=label)
        else:
            self.assert_field_compatible(out)
            out.data[:] = func(self.data)
            if label:
                out.label = label
            return out
        
        
    @abstractproperty
    def integral(self) -> Union[np.ndarray, float]: pass 
    @abstractmethod
    def to_scalar(self, scalar: Union[str, int] = 'norm',
                  label: Optional[str] = None) -> "ScalarField": pass
                  
                  
    @property
    def average(self) -> Union[np.ndarray, float]:
        """ determine the average of data
        
        This is calculated by integrating each component of the field over space
        and dividing by the grid volume
        """
        return self.integral / self.grid.volume
    

    @property    
    def fluctuations(self):
        """ :class:`numpy.ndarray`: fluctuations over the entire space.
        
        The fluctuations are defined as the standard deviation of the data
        scaled by the cell volume. This definition makes the fluctuations
        independent of the discretization. It corresponds to the physical
        scaling available in the :func:`~DataFieldBase.random_normal`.
        
        Returns:
            :class:`numpy.ndarray`: A tensor with the same rank of the field,
            specifying the fluctuations of each component of the tensor field
            individually. Consequently, a simple scalar is returned for a
            :class:`~pde.fields.scalar.ScalarField`.
        """
        scaled_data = self.data * np.sqrt(self.grid.cell_volume_data)
        axes = tuple(range(self.rank, self.data.ndim))
        return np.std(scaled_data, axis=axes)
            
    
    @property
    def magnitude(self) -> float:
        """ float: determine the magnitude of the field.
        
        This is calculated by getting a scalar field using the default arguments
        of the :func:`to_scalar` method and averaging the result over the whole
        grid.
        """
        if self.rank == 0:
            return float(self.average)
        elif self.rank > 0:
            return self.to_scalar().average
        else:
            raise NotImplementedError('Magnitude cannot be determined for '
                                      'field ' + self.__class__.__name__)
            
            
    def smooth(self: T2, sigma: Optional[float] = 1, out: Optional[T2] = None,
               label: str = None) -> T2:
        """ applies Gaussian smoothing with the given standard deviation

        This function respects periodic boundary conditions of the underlying
        grid, using reflection when no periodicity is specified.
        
        sigma (float, optional):
            Gives the standard deviation of the smoothing in real length units
            (default: 1)
        out (FieldBase, optional):
            Optional field into which the smoothed data is stored
        label (str, optional):
            Name of the returned field

        Returns:
            Field with smoothed data. This is stored at `out` if given.             
        """
        # allocate memory for storing output
        data_in = self._data
        if out is None:
            out = self.__class__(self.grid, label=self.label)
        else:
            self.assert_field_compatible(out)
        
        # apply Gaussian smoothing for each axis
        data_out = out._data
        for axis in range(-len(self.grid.axes), 0):
            sigma_dx = sigma / self.grid.discretization[axis]
            mode = 'wrap' if self.grid.periodic[axis] else 'reflect'
            ndimage.gaussian_filter1d(data_in, sigma=sigma_dx, axis=axis,
                                      output=data_out, mode=mode)
            data_in = data_out
            
        # return the data in the correct field class
        if label:
            out.label = label
        return out
            
            
    def plot_line(self, 
                  extract: str = 'auto',
                  ylabel: str = None,
                  ax=None,
                  title: str = None,
                  show: bool = False,
                  **kwargs):
        r""" visualize a field using a 1d cut
        
        Args:
            extract (str): The method used for extracting the line data.
            ylabel (str): Label of the y-axis. If omitted, the label is chosen
                automatically from the data field.
            ax: Figure axes to be used for plotting. If `None`, a new figure is
                created
            title (str): Title of the plot.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Additional keyword arguments are passed to
                `matplotlib.pyplot.plot`
                
        Returns:
            Instance of the line returned by `plt.plot`
        """
        import matplotlib.pyplot as plt
        # obtain data
        line_data = self.get_line_data(extract=extract)
        
        # plot the data (either using pyplot or the supplied axes)
        if ax is None:
            line, = plt.plot(line_data['data_x'], line_data['data_y'], **kwargs)
            ax = plt.gca()
        else:
            line, = ax.plot(line_data['data_x'], line_data['data_y'], **kwargs)

        # set some default properties
        ax.set_xlabel(line_data['label_x'])
        if ylabel is None:
            ylabel = line_data.get('label_y', self.label)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        from ..visualization.plotting import finalize_plot
        finalize_plot(ax, title=title, show=show)
            
        return line
    
    
    def plot_image(self, ax=None,
                   colorbar: bool = False,
                   transpose: bool = False,
                   title: Optional[str] = None,
                   show: bool = False,
                   **kwargs):
        r""" visualize an 2d image of the field

        Args:
            ax:
                Figure axes to be used for plotting. If `None`, a new figure is
                created
            colorbar (bool):
                Determines whether a colorbar is shown
            transpose (bool):
                Determines whether the transpose of the data should is plotted
            title (str):
                Title of the plot. If omitted, the title is chosen automatically
                based on the label the data field.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs:
                Additional keyword arguments are passed to
                :func:`matplotlib.pyplot.imshow`.
                
        Returns:
            Result of :func:`matplotlib.pyplot.imshow`
        """
        import matplotlib.pyplot as plt
        
        # obtain image data
        get_image_args = {}
        # FIXME: rename scalar_method to method
        for key in ['performance_goal', 'scalar_method']:
            if key in kwargs:
                get_image_args[key] = kwargs.pop(key)
        img = self.get_image_data(**get_image_args)
        
        if transpose:
            # adjust image data such that the transpose is plotted
            img['data'] = img['data'].T
            img['label_x'], img['label_y'] = img['label_y'], img['label_x']
        
        # plot the image (either using pyplot or the supplied axes)
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('interpolation', 'none')
        if ax is None:
            ax = plt.figure().gca()
        res = ax.imshow(img['data'], extent=img['extent'], **kwargs)
            
        # set some default properties
        ax.set_xlabel(img['label_x'])
        ax.set_ylabel(img['label_y'])
        if title is None:
            title = img.get('title', self.label)

        if colorbar:
            from ..tools.misc import add_scaled_colorbar
            add_scaled_colorbar(res, ax=ax)

        from ..visualization.plotting import finalize_plot
        finalize_plot(ax, title=title, show=show)
            
        return res


    def plot_vector(self,
                    method: str = 'quiver',
                    ax=None,
                    transpose: bool = False,
                    title: Optional[str] = None,
                    show: bool = False,
                    **kwargs):
        r""" visualize a 2d vector field

        Args:
            method (str): Plot type that is used. This can be either `quiver`
                or `streamplot`.
            ax: Figure axes to be used for plotting. If `None`, a new figure is
                created
            transpose (bool): determines whether the transpose of the data
                should be plotted.
            title (str): Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            \**kwargs: Additional keyword arguments are passed to
                :func:`matplotlib.pyplot.quiver` or
                :func:`matplotlib.pyplot.streamplot`.
                
        Returns:
            Result of `plt.quiver` or `plt.streamplot`
        """
        import matplotlib.pyplot as plt
        
        # obtain image data
        get_image_args = {}
        # FIXME: rename scalar_method to method
        for key in ['performance_goal', 'scalar_method']:
            if key in kwargs:
                get_image_args[key] = kwargs.pop(key)
        data = self.get_vector_data(**get_image_args)
        
        if transpose:
            # adjust image data such that the transpose is plotted
            data['x'], data['y'] = data['y'], data['x']
            data['data_x'], data['data_y'] = data['data_y'].T, data['data_x'].T
            data['label_x'], data['label_y'] = data['label_y'], data['label_x']
        
        # determine which plotting function to use
        if method == 'quiver':
            plot_func = plt.quiver if ax is None else ax.quiver
            
        elif method == 'streamplot':
            plot_func = plt.streamplot if ax is None else ax.streamplot
                
        else:
            raise ValueError(f'Vector plot `{method}` is not supported.')

        # do the actual plotting
        res = plot_func(data['x'], data['y'], data['data_x'], data['data_y'],
                        **kwargs)        
        if ax is None:                
            ax = plt.gca()
            
        # set some default properties
        ax.set_aspect('equal')
        ax.set_xlabel(data['label_x'])
        ax.set_ylabel(data['label_y'])
        if title is None:
            title = data.get('title', self.label)

        from ..visualization.plotting import finalize_plot
        finalize_plot(ax, title=title, show=show)
            
        return res
        
    