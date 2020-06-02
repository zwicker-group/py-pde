'''
Defines base classes


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import functools
import operator
import logging
import json
import warnings
from pathlib import Path 
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import (Tuple, Callable, Optional, Union, Any, Dict, TypeVar,
                    TYPE_CHECKING)

import numpy as np
import numba as nb        
from scipy import interpolate, ndimage

from ..grids.base import (GridBase, discretize_interval, DimensionError,
                          DomainError)
from ..grids.cartesian import CartesianGridBase
from ..grids.boundaries.axes import BoundariesData
from ..tools.numba import jit, address_as_void_pointer
from ..tools.cache import cached_method
from ..tools.docstrings import fill_in_docstring


if TYPE_CHECKING:
    from .scalar import ScalarField  # @UnusedImport



class PlotReference():
    """ contains all information to update a plot element """
    
    __slots__ = ['ax', 'element', 'parameters']
    
    def __init__(self, ax, element: Any, parameters: Dict[str, Any] = None):
        """
        Args:
            ax (:class:`matplotlib.axes.Axes`): The axes of the element
            element (:class:`matplotlib.artist.Artist`): The actual element 
            parameters (dict): Parameters to recreate the plot element
        """
        self.ax = ax
        self.element = element
        self.parameters = {} if parameters is None else parameters
        


ArrayLike = Union[np.ndarray, float]
OptionalArrayLike = Optional[ArrayLike]
TField = TypeVar('TField', bound='FieldBase')



class FieldBase(metaclass=ABCMeta):
    """ abstract base class for describing (discretized) fields
    
    Attributes:
        label (str):
            Name of the field
    """ 
    
    _subclasses: Dict[str, 'FieldBase'] = {}  # all classes inheriting from this
    readonly = False
    
    
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
        self._grid = grid
        self._data: np.ndarray = data
        self.label = label
        self._logger = logging.getLogger(self.__class__.__name__)


    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls


    @classmethod
    def from_state(cls, attributes: Dict[str, Any],
                   data: np.ndarray = None) -> "FieldBase":
        """ create a field from given state.
        
        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`numpy.ndarray`, optional):
                Data values at the support points of the grid defining the field
        """
        # base class was chosen => select correct class from attributes
        class_name = attributes.pop('class')

        if class_name == cls.__name__:
            raise RuntimeError('Cannot reconstruct abstract class' 
                               f'`{class_name}`')
        
        # call possibly overwritten classmethod from subclass
        return cls._subclasses[class_name].from_state(attributes, data)

            
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
                assert json.loads(fp.attrs['class']) == 'FieldCollection'
                obj = FieldCollection._from_hdf_dataset(fp)
                
            elif len(fp) == 1:
                # a single field is stored in the data
                dataset = fp[list(fp.keys())[0]]  # retrieve only dataset
                obj = cls._from_hdf_dataset(dataset)  # type: ignore
                
            else:
                raise RuntimeError('Multiple data fields were found in the '
                                   'file but no FieldCollection is expected')
        return obj
                
                
    @classmethod
    def _from_hdf_dataset(cls, dataset) -> "FieldBase":
        """ construct a field by reading data from an hdf5 dataset """
        # copy attributes from hdf
        attributes = dict(dataset.attrs)
        
        # determine class
        class_name = json.loads(attributes.pop('class'))
        field_cls = cls._subclasses[class_name]
        
        # unserialize the attributes
        attributes = field_cls.unserialize_attributes(attributes)
        return field_cls.from_state(attributes, data=dataset)

    
    @property
    def grid(self) -> GridBase:
        """ GridBase: The grid on which the field is defined """
        return self._grid


    def to_file(self, filename: str, **kwargs):
        r""" store field in a file
        
        The extension of the filename determines what format is being used. If
        it ends in `.h5` or `.hdf`, the Hierarchical Data Format is used. The
        other supported format are images, where only the most typical formats
        are supported. 
        
        Args:
            filename (str):
                Path where the data is stored
            \**kwargs:
                Additional parameters may be supported for some formats 
        """
        extension = Path(filename).suffix.lower()
        
        if extension in {'.hdf', '.hdf5', '.he5', '.h5'}:
            import h5py
            with h5py.File(filename, "w") as fp:
                self._write_hdf_dataset(fp, **kwargs)
                
        elif extension in {'.png', '.jpg', '.jpeg', '.tif', '.pdf', '.svg'}:
            self._write_to_image(filename, **kwargs)
            
        else:
            raise ValueError('Do not know how to save data to file with '
                             f'extensions `{extension}`')


    def _write_hdf_dataset(self, hdf_path, key: str = 'data'):
        """ write data to a given hdf5 path `hdf_path` """
        # write the data
        dataset = hdf_path.create_dataset(key, data=self.data)

        # write attributes        
        for key, value in self.attributes_serialized.items():
            dataset.attrs[key] = value


    def _write_to_image(self, filename: str, **kwargs):
        """ write data to image 
        
        Args:
            filename (str): The path to the image that will be created
        """
        raise NotImplementedError(f'Cannot save {self.__class__.__name__} as '
                                  'an image')

    
    @abstractmethod
    def copy(self: TField, data: OptionalArrayLike = None, label: str = None) \
        -> TField: pass
            
         
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
    def data(self) -> np.ndarray:
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
    def attributes(self) -> Dict[str, Any]:
        """ dict: describes the state of the instance (without the data) """
        return {'class': self.__class__.__name__,
                'grid': self.grid,
                'label': self.label}
    

    @property
    def attributes_serialized(self) -> Dict[str, str]:
        """ dict: serialized version of the attributes """
        results = {}
        for key, value in self.attributes.items():
            if key == 'grid':
                results[key] = value.state_serialized
            else:
                results[key] = json.dumps(value)
        return results
    
    
    @classmethod
    def unserialize_attributes(cls, attributes: Dict[str, str]) \
            -> Dict[str, Any]:
        """ unserializes the given attributes
        
        Args:
            attributes (dict):
                The serialized attributes
                
        Returns:
            dict: The unserialized attributes
        """
        # base class was chosen => select correct class from attributes
        class_name = json.loads(attributes['class'])

        if class_name == cls.__name__:
            raise RuntimeError('Cannot reconstruct abstract class' 
                               f'`{class_name}`')
        
        # call possibly overwritten classmethod from subclass
        return cls._subclasses[class_name].unserialize_attributes(attributes)


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
        """ test fields for equality, ignoring the label """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.grid == other.grid and
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
    

    @abstractmethod
    def get_line_data(self, scalar: str = 'auto', extract: str = 'auto'): pass

    @abstractmethod
    def get_image_data(self): pass
    
    @abstractmethod
    def plot(self,
             kind: str = 'auto',
             title: str = None,
             filename: str = None,
             show: bool = True,
             close_figure: bool = False, **kwargs): pass
    
    @abstractmethod
    def update_plot(self, reference): pass
    


TDataField = TypeVar('TDataField', bound='DataFieldBase')


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
                Field values at the support points of the grid. The data is
                copied from the supplied array.
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
        
        These values are uncorrelated in space.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            vmin (float):
                Smallest possible random value
            vmax (float):
                Largest random value
            label (str, optional):
                Name of the field
            seed (int, optional):
                Seed of the random number generator. If `None`, the current
                state is not changed.
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
        
        These values are uncorrelated in space.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            mean (float):
                Mean of the Gaussian distribution
            std (float):
                Standard deviation of the Gaussian distribution
            scaling (str):
                Determines how the values are scaled. Possible choices are
                'none' (values are drawn from a normal distribution with
                given mean and standard deviation) or 'physical' (the variance
                of the random number is scaled by the inverse volume of the grid
                cell; this is useful for physical quantities, which vary less in
                larger volumes).            
            label (str, optional):
                Name of the field
            seed (int, optional):
                Seed of the random number generator. If `None`, the current
                state is not changed.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if scaling == 'none':
            noise_scale = std
        elif scaling == 'physical':
            noise_scale = std / np.sqrt(grid.cell_volumes)
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
        
        The resulting fields will be highly correlated in space and can thus
        serve for testing differential operators. 
        
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
            modes (int):
                Number :math:`M` of harmonic modes
            harmonic (callable):
                Determines which harmonic function is used. Typical values are
                :func:`numpy.sin` and :func:`numpy.cos`, which basically relate
                to different boundary conditions applied at the grid boundaries.
            axis_combination (callable):
                Determines how values from different axis are combined. Typical 
                choices are :func:`numpy.multiply` and :func:`numpy.add`
                resulting in products and sums of the values along axes,
                respectively.
            label (str, optional):
                Name of the field
            seed (int, optional):
                Seed of the random number generator. If `None`, the current
                state is not changed.
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
        r""" create a field of random values with colored noise
                
        The spatially correlated values obey
        
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
            label (str, optional):
                Name of the field
            seed (int, optional):
                Seed of the random number generator. If `None`, the curren
                 state is not changed.
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
    def from_state(cls, attributes: Dict[str, Any],
                   data: np.ndarray = None) -> "DataFieldBase":
        """ create a field from given state.
        
        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`numpy.ndarray`, optional):
                Data values at the support points of the grid defining the field
        """
        if 'class' in attributes:
            class_name = attributes.pop('class')
            assert class_name == cls.__name__
            
        # create the instance from the attributes
        return cls(attributes.pop('grid'), data=data, **attributes)
        
        
    def copy(self: TDataField,
             data: OptionalArrayLike = None,
             label: str = None) -> TDataField:
        """ return a copy of the data, but not of the grid
        
        Args:
            data (:class:`numpy.ndarray`, optional):
                Data values at the support points of the grid that define the
                field.
            label (str, optional):
                Name of the copied field
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
    
    
    @classmethod
    def unserialize_attributes(cls, attributes: Dict[str, str]) \
            -> Dict[str, Any]:
        """ unserializes the given attributes
        
        Args:
            attributes (dict):
                The serialized attributes
                
        Returns:
            dict: The unserialized attributes
        """
        results = {}
        for key, value in attributes.items():
            if key == 'grid':
                results[key] = GridBase.from_state(value)
            else:
                results[key] = json.loads(value)
        return results
    
        
    def _write_to_image(self, filename: str, **kwargs):
        r""" write data to image 
        
        Args:
            filename (str):
                The path to the image that will be created
            \**kwargs:
                Additional keyword arguments that affect the image. For
                instance, some fields support a `scalar` argument that
                determines how they are converted to a scalar. Non-Cartesian
                grids might support a `performance_goal` argument to influence
                how an image is created from the raw data. Finally, the
                remaining arguments are are passed to
                :func:`matplotlib.pyplot.imsave` to affect the appearance.
        """
        import matplotlib.pyplot as plt
        # obtain image data
        get_image_args = {}
        for key in ['performance_goal', 'scalar']:
            if key in kwargs:
                get_image_args[key] = kwargs.pop(key)
        img = self.get_image_data(**get_image_args)
        
        kwargs.setdefault('cmap', 'gray')
        plt.imsave(filename, img['data'], origin='lower', **kwargs)
    

    def _make_interpolator_scipy(self, method: str = 'linear',
                                 fill: float = None, **kwargs) -> Callable:
        r""" returns a function that can be used to interpolate values.
        
        This uses scipy.interpolate.RegularGridInterpolator and the
        interpolation method can thus be chosen when calling the returned
        function using the `method` keyword argument. Keyword arguments are
        directly forwarded to the constructor of `RegularGridInterpolator`.
        
        Note that this interpolator does not respect periodic boundary
        conditions, yet.
        
        Args:
            method (str):
                The method used for interpolation. If 'linear' or 'nearest'
                :class:`scipy.interpolate.RegularGridInterpolator` is used with
                the specified method. If 'rbf', :class:`scipy.interpolate.Rbf`
                is used for radial basis function interpolation.
            fill (float, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
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
            
        # set the fill behavior
        if fill is None:
            kwargs['bounds_error'] = True
        else:
            kwargs['bounds_error'] = False
            kwargs['fill_value'] = fill
            
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


    @fill_in_docstring
    def _make_interpolator_compiled(self, bc: BoundariesData = 'natural',
                                    fill: float = None) -> Callable:
        """ return a compiled interpolator
        
        This interpolator respects boundary conditions and can thus interpolate
        values in the whole grid volume. However, close to corners, the
        interpolation might not be optimal, in particular for periodic grids.
        
        Args:
            bc:
                The boundary conditions applied to the field. {ARG_BOUNDARIES}
            fill (float, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
                
        Returns:
            A function which returns interpolated values when called with
            arbitrary positions within the space of the grid. 
        """
        grid = self.grid
        grid_dim = len(grid.axes)
        data_shape = self.data_shape

        # create an interpolator for a single point
        if fill is not None:
            if self.rank == 0:
                fill = float(fill)
            else:
                fill = np.broadcast_to(fill, self.data_shape).astype(float)
        interpolate_single = \
            grid.make_interpolator_compiled(bc=bc, rank=self.rank, fill=fill)
        
        # extract information about the data field
        data_addr = self.data.ctypes.data
        shape, dtype = self.data.shape, self.data.dtype
        
        @jit
        def interpolator(point: np.ndarray, data: np.ndarray = None) \
                -> np.ndarray:
            """ return the interpolated value at the position `point`
             
            Args:
                point (:class:`numpy.ndarray`):
                    The list of points. This point coordinates should be given
                    along the last axis, i.e., the shape should be `(..., dim)`.
                data (:class:`numpy.ndarray`, optional):
                    The discretized field values. If omitted, the data of the 
                    current field is used, which should be the default. However,
                    this option can be useful to interpolate other fields
                    defined on the same grid without recreating the interpolator 
                     
            Returns:
                :class:`numpy.ndarray`: The interpolated values at the points
            """
            # check input
            point = np.atleast_1d(point)
            if point.shape[-1] != grid_dim:
                raise DimensionError('Dimension of the interpolation point '
                                     'does not match grid dimension')
            point_shape = point.shape[:-1]
            
            # reconstruct data field from memory address
            if data is None:
                data = nb.carray(address_as_void_pointer(data_addr), shape,
                                 dtype)
             
            # interpolate at every point
            out = np.empty(data_shape + point_shape)
            for idx in np.ndindex(point_shape):
                out[(...,) + idx] = interpolate_single(data, point[idx])
             
            return out

        return interpolator  # type: ignore


    @cached_method()
    def make_interpolator(self, method: str = 'numba',
                          fill: float = None,
                          **kwargs) -> Callable:
        r""" returns a function that can be used to interpolate values.
        
        Args:
            method (str):
                Determines the method being used for interpolation.
                Possible values are:
                
                * `scipy_nearest`: Use scipy to interpolate to nearest neighbors
                * `scipy_linear`: Linear interpolation using scipy
                * `numba`: Linear interpolation using numba (default)

            fill (float, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            \**kwargs:
                Additional keyword arguments are passed to the individual
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
            return self._make_interpolator_scipy(method=method[6:], fill=fill,
                                                 **kwargs)
        elif method == 'numba':
            return self._make_interpolator_compiled(fill=fill, **kwargs)
        else:
            raise ValueError(f'Unknown interpolation method `{method}`')
            

    def interpolate(self, point, method: str = 'numba', fill: float = None,
                    **kwargs):
        r""" interpolate the field to points between support points
        
        Args:
            point (:class:`numpy.ndarray`):
                The points at which the values should be obtained. This is given
                in grid coordinates.
            method (str):
                Determines the method being used for interpolation.
            fill (float, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            \**kwargs:
                Additional keyword arguments are forwarded to the method
                :meth:`DataFieldBase.make_interpolator`.
                
        Returns:
            :class:`numpy.ndarray`: the values of the field
        """
        point = np.asarray(point)
        return self.make_interpolator(method=method, fill=fill, **kwargs)(point)


    def interpolate_to_grid(self: TDataField, grid: GridBase,
                            method: str = 'numba',
                            fill: float = None,
                            label: Optional[str] = None) -> TDataField:
        """ interpolate the data of this field to another grid.
        
        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            method (str):
                Specifies interpolation method, e.g., 'numba', 'scipy_linear',
                'scipy_nearest' .
            fill (float, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            label (str, optional):
                Name of the returned field
            
        Returns:
            Field of the same rank as the current one.
        """
        if self.grid.dim != grid.dim:
            raise DimensionError(f'Grid dimensions are incompatible '
                                 f'({self.grid.dim:d} != {grid.dim:d})')
            
        # determine the points at which data needs to be calculated
        if isinstance(grid, CartesianGridBase):
            # convert to a Cartesian grid
            points = self.grid.point_from_cartesian(grid.cell_coords)
            
        elif self.grid.__class__ is grid.__class__:
            # convert within the same grid class
            points = grid.cell_coords
            
        else:
            # this type of interpolation is not supported
            raise NotImplementedError('Cannot convert '
                                      f'{self.grid.__class__.__name__} to '
                                      f'{grid.__class__.__name__}')
        
        # interpolate the data to the grid
        data = self.interpolate(points, method, fill)
        return self.__class__(grid, data, label=label)
    
    
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
            raise DimensionError(f'Dimension mismatch for point {point}')
        
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
            raise DomainError('Point lies outside grid')
    
        # alter each point in second iteration
        for coords, weight in cells:
            chng = weight * amount / (total_weight * grid.cell_volumes[coords])
            self.data[(Ellipsis,) + coords] += chng                                                  
        
        
    @fill_in_docstring
    def get_boundary_values(self, axis: int,
                            upper: bool,
                            bc: BoundariesData = 'natural') -> np.ndarray:
        """ get the field values directly on the specified boundary 
        
        Args:
            axis (int):
                The axis perpendicular to the boundary
            upper (bool):
                Whether the boundary is at the upper side of the axis 
            bc:
                The boundary conditions applied to the field. {ARG_BOUNDARIES}
            
        Returns:
            :class:`~numpy.ndarray`: The discretized values on the boundary
        """
        interpolator = self.make_interpolator(bc=bc)
        points = self.grid._boundary_coordinates(axis, upper)
        return interpolator(points)
        
        
    @fill_in_docstring
    def make_get_boundary_values(self, axis: int,
                                 upper: bool,
                                 bc: BoundariesData = 'natural') -> np.ndarray:
        """ make a function calculating field values on the specified boundary 
        
        Args:
            axis (int):
                The axis perpendicular to the boundary
            upper (bool):
                Whether the boundary is at the upper side of the axis 
            bc:
                The boundary conditions applied to the field. {ARG_BOUNDARIES}
            
        Returns:
            callable: A function returning the values on the boundary. The
            function has the signature `(data=None, out=None)`, which allows
            specifying an input and an output :class:`~numpy.ndarray`. If `data`
            is omitted, the data of the current field is used. The resulting
            interpolation is written to `out` if it is present. Otherwise, a new
            array is created.
        """
        interpolator = self.make_interpolator(bc=bc)
        points = self.grid._boundary_coordinates(axis, upper)
        
        # TODO: use jit_allocated_out with pre-calculated shape
        
        @jit
        def get_boundary_values(data: np.ndarray = None,
                                out: np.ndarray = None):
            """ interpolate the field at the boundary
            
            Args:
                data (:class:`~numpy.ndarray`, optional):
                    The data values that are used for interpolation. The data of
                    the current field is used if `data = None`.
                out (:class:`~numpy.ndarray`, optional):
                    The array into which the interpolated results are written. A
                    new array is created if `out = None`.
                    
            Returns:
                :class:`numpy.ndarray`: The interpolated values on the boundary.
            """
            res = interpolator(points, data)
            if out is None:
                return res
            else:
                # the following just copies the data from res to out. It is a
                # workaround for a bug in numba existing up to at least ver 0.49
                out[...] = res[()]
                return out
            
        return get_boundary_values
        
        
    def apply(self: TDataField, func: Callable,
              out: Optional[TDataField] = None,
              label: str = None) -> TDataField:
        """ applies a function to the data and returns it as a field
        
        Args:
            func (callable):
                The (vectorized) function being applied to the data.
            out (FieldBase, optional):
                Optional field into which the data is written
            label (str, optional):
                Name of the returned field

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
    def to_scalar(self, scalar: str = 'auto',
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
        scaled_data = self.data * np.sqrt(self.grid.cell_volumes)
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
            
            
    def smooth(self: TDataField, sigma: Optional[float] = 1,
               out: Optional[TDataField] = None,
               label: str = None) -> TDataField:
        """ applies Gaussian smoothing with the given standard deviation

        This function respects periodic boundary conditions of the underlying
        grid, using reflection when no periodicity is specified.
        
        sigma (float, optional):
            Gives the standard deviation of the smoothing in real length units
            (default: 1)
        out (FieldBase, optional):
            Optional field into which the smoothed data is stored. Setting this
            to the input field enables in-place smoothing.
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


    def get_line_data(self, scalar: str = 'auto', extract: str = 'auto') \
            -> Dict[str, Any]:
        """ return data for a line plot of the field
        
        Args:
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            extract (str):
                The method used for extracting the line data. See the docstring
                of the grid method `get_line_data` to find supported values.
        
        Returns:
            dict: Information useful for performing a line plot of the field
        """
        # turn field into scalar field
        scalar_data = self.to_scalar(scalar).data
        
        # extract the line data
        data = self.grid.get_line_data(scalar_data, extract=extract)
        if 'label_y' in data and data['label_y']:
            if self.label:
                data['label_y'] = f"{self.label} ({data['label_y']})"
        else:
            data['label_y'] = self.label
        return data  # type: ignore  
        
                
    def get_image_data(self, scalar: str = 'auto',
                       transpose: bool = False,
                       **kwargs) -> Dict[str, Any]:
        r""" return data for plotting an image of the field
        
        Args:
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            transpose (bool):
                Determines whether the transpose of the data should is plotted
            \**kwargs:
                Additional parameters are forwarded to `grid.get_image_data`
        
        Returns:
            dict: Information useful for plotting an image of the field        
        """
        # turn field into scalar field
        scalar_data = self.to_scalar(scalar).data
        
        # extract the image data
        data = self.grid.get_image_data(scalar_data, **kwargs)  # type: ignore 
        data['title'] = self.label
        
        if transpose:
            # adjust image data such that the transpose is plotted
            data['data'] = data['data'].T
            data['label_x'], data['label_y'] = data['label_y'], data['label_x']
            
        return data  # type: ignore  

    
    def get_vector_data(self, **kwargs) -> Dict[str, Any]:
        r""" return data for a vector plot of the field
        
        Args:
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`
        
        Returns:
            dict: Information useful for plotting an vector field        
        """
        raise NotImplementedError()
    
                
    def _plot_line(self, ax,
                   extract: str = 'auto',
                   ylabel: str = None,
                   **kwargs) -> PlotReference:
        r""" visualize a field using a 1d line plot
        
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            extract (str):
                The method used for extracting the line data.
            ylabel (str):
                Label of the y-axis. If omitted, the label is chosen 
                automatically from the data field.
            \**kwargs:
                Additional keyword arguments are passed to
                :func:`matplotlib.pyplot.plot`
                
        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # obtain data for the plot
        line_data = self.get_line_data(extract=extract)
        
        # do the plot
        line2d, = ax.plot(line_data['data_x'], line_data['data_y'],
                              **kwargs)

        # set some default properties
        ax.set_xlabel(line_data['label_x'])
        if ylabel is None:
            ylabel = line_data.get('label_y', self.label)
        if ylabel:
            ax.set_ylabel(ylabel)
                
        return PlotReference(ax, line2d, {'extract': extract})
    
    
    def _update_line_plot(self, reference: PlotReference) -> None:
        """ update a line plot with the current field values
        
        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        import matplotlib as mpl
        
        # obtain data for the plot
        extract = reference.parameters.get('extract', 'auto')
        line_data = self.get_line_data(extract=extract)
        
        line2d = reference.element
        if isinstance(line2d, mpl.lines.Line2D):
            # update old plot
            line2d.set_xdata(line_data['data_x'])
            line2d.set_ydata(line_data['data_y'])
            
        else:
            raise ValueError(f'Unsupported plot reference {reference}')


    def _plot_image(self, ax,
                    colorbar: bool = False,
                    scalar: str = 'auto',
                    transpose: bool = False,
                    **kwargs) -> PlotReference:
        r""" visualize a field using a 2d density plot

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            colorbar (bool):
                Determines whether a colorbar is shown
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            transpose (bool):
                Determines whether the transpose of the data should is plotted
            \**kwargs:
                Additional keyword arguments that affect the image. For
                instance, some fields support a `scalar` argument that
                determines how they are converted to a scalar. Non-Cartesian
                grids might support a `performance_goal` argument to influence
                how an image is created from the raw data. Finally, the
                remaining arguments are are passed to
                :func:`matplotlib.pyplot.imshow` to affect the appearance.
                
        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # obtain image data
        data = self.get_image_data(scalar, transpose)
        
        if ax is None:
            # create new figure
            import matplotlib.pyplot as plt
            ax = plt.subplots()[1]
            
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('interpolation', 'none')
        axes_image = ax.imshow(data['data'], extent=data['extent'], **kwargs)
            
        # set some default properties
        ax.set_xlabel(data['label_x'])
        ax.set_ylabel(data['label_y'])
        ax.set_title(data.get('title', self.label))

        if colorbar:
            from ..tools.misc import add_scaled_colorbar
            add_scaled_colorbar(axes_image, ax=ax)
                
        parameters = {'scalar': scalar, 'transpose': transpose}
        return PlotReference(ax, axes_image, parameters)

    
    def _update_image_plot(self, reference: PlotReference) -> None:
        """ update an image plot with the current field values
        
        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        # obtain image data
        p = reference.parameters
        data = self.get_image_data(scalar=p.get('scalar', 'auto'),
                                   transpose=p.get('transpose', False))
        
        # update the axes image
        reference.element.set_array(data['data'])


    def _plot_vector(self, ax,
                     method: str = 'quiver',
                     transpose: bool = False,
                     max_points: int = 16,
                     **kwargs) -> PlotReference:
        r""" visualize a field using a 2d vector plot

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            method (str):
                Plot type that is used. This can be either `quiver` or
                `streamplot`.
            transpose (bool):
                Determines whether the transpose of the data should be plotted.
            max_points (int):
                The maximal number of points that is used along each axis. This
                argument is only used for quiver plots.
            \**kwargs:
                Additional keyword arguments are passed to
                :func:`matplotlib.pyplot.quiver` or
                :func:`matplotlib.pyplot.streamplot`.
                
        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # do the plotting using the chosen method
        if method == 'quiver':
            data = self.get_vector_data(transpose=transpose,
                                        max_points=max_points)
            element = ax.quiver(data['x'], data['y'], data['data_x'],
                                data['data_y'], **kwargs)        
            
        elif method == 'streamplot':
            data = self.get_vector_data(transpose=transpose)
            element = ax.streamplot(data['x'], data['y'], data['data_x'],
                                    data['data_y'], **kwargs)        
                
        else:
            raise ValueError(f'Vector plot `{method}` is not supported.')
            
        # set some default properties of the plot
        ax.set_aspect('equal')
        ax.set_xlabel(data['label_x'])
        ax.set_ylabel(data['label_y'])
        ax.set_title(data.get('title', self.label))

        parameters = {'method': method, 'transpose': transpose,
                      'max_points': max_points, 'kwargs': kwargs}
        return PlotReference(ax, element, parameters)

    
    def _update_vector_plot(self, reference: PlotReference) -> None:
        """ update a vector plot with the current field values
        
        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        method = reference.parameters.get('method', 'quiver')
        transpose = reference.parameters.get('transpose', False)
        if method == 'quiver':
            max_points = reference.parameters.get('max_points')
            data = self.get_vector_data(transpose=transpose,
                                        max_points=max_points)
            reference.element.set_UVC(data['data_x'], data['data_y'])
            
        elif method == 'streamplot':
            ax = reference.ax
            kwargs = reference.parameters.get('kwargs', {})
            data = self.get_vector_data(transpose=transpose)
            # remove old streamplot
            reference.element.remove()  
            # update with new streamplot
            reference.element = ax.streamplot(data['x'], data['y'],
                                              data['data_x'], data['data_y'],
                                              **kwargs)        
                
        else:
            raise ValueError(f'Vector plot `{method}` is not supported.')


    def plot(self,
             kind: str = 'auto',
             title: str = None,
             filename: str = None,
             show: bool = True,
             close_figure: bool = False,
             ax=None,
             **kwargs):
        r""" visualize the field
        
        Args:
            kind (str):
                Determines the visualizations. Supported values are `image`, 
                `line`, `vector`, or `interactive`. Alternatively, `auto`
                determines the best visualization based on the field itself.
            title (str):
                Title of the plot. If omitted, the title is chosen
                automatically based on the label the data field.
            filename (str, optional):
                If given, the plot is written to the specified file.
            show (bool):
                Flag setting whether :func:`matplotlib.pyplot.show` is called
            close_figure (bool):
                Flag setting whether the figure is closed (after it was shown)
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting. If `None`, a new figure is
                created. This has no effect if a `reference` is supplied.
            \**kwargs:
                All additional keyword arguments are forwarded to the actual
                plotting functions.
                
        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        import matplotlib.pyplot as plt
        from ..visualization.contexts import disable_interactive
        
        # pre-process the kinds
        if kind == 'auto':
            # determine best plot for this field
            if (isinstance(self, DataFieldBase) and self.rank == 1 and
                    self.grid.dim == 2):
                kind = 'vector'
            elif len(self.grid.shape) == 1:
                kind = 'line'
            else:
                kind = 'image'
                
        elif kind == 'quiver':
            kind = 'vector'
            kwargs['method'] = 'quiver'

        elif kind == 'streamplot':
            kind = 'vector'
            kwargs['method'] = 'streamplot'

        # disable interactive plotting temporarily
        with disable_interactive():
            
            if ax is None:
                # create new figure
                ax = plt.subplots()[1]
            
            # do the actual plotting
            if kind == 'image':
                reference = self._plot_image(ax=ax, **kwargs)
            elif kind == 'line':
                reference = self._plot_line(ax=ax, **kwargs)
            elif kind == 'vector':
                reference = self._plot_vector(ax=ax, **kwargs)
            else:
                raise ValueError(f'Unsupported plot `{kind}`. Possible choices '
                                 'are `image`, `line`, `vector`, or `auto`.')

            # obtain figure from axes
            fig = reference.ax.get_figure()  
    
            # finishing touches...            
            if title is not None:
                reference.ax.set_title(title)
            if filename:
                fig.savefig(filename)
                
        # decide what to do with the final plot
        if show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.show()
        if close_figure:
            plt.close(fig)        
            
        return reference

    
    def update_plot(self, reference: PlotReference) -> None:
        """ update a plot with the current field values
        
        Args:
            reference (:class:`PlotReference`):
                The reference to the plot to updated
        """
        import matplotlib as mpl
        
        # update the plot based on the given reference
        el = reference.element
        if isinstance(el, mpl.lines.Line2D):
            self._update_line_plot(reference)
        elif isinstance(el, mpl.image.AxesImage):
            self._update_image_plot(reference)
        elif isinstance(el, (mpl.quiver.Quiver, mpl.streamplot.StreamplotSet)):
            self._update_vector_plot(reference)
        else:
            raise ValueError(f'Unknown plot element {el.__class__.__name__}')
    
    
    def plot_interactive(self, scalar: str = 'auto', **kwargs):
        """ create an interactive plot of the field using :mod:`napari`
        
        Args:
            scalar (str): The method for obtaining scalar values of fields        
            **kwargs: Extra arguments are passed to :class:`napari.Viewer`
        """
        from ..visualization.plotting import napari_viewer
        
        with napari_viewer(self.grid, **kwargs) as viewer:
            viewer.add_image(self.to_scalar(scalar).data,
                             name=self.label,
                             rgb=False,
                             scale=self.grid.discretization)
    