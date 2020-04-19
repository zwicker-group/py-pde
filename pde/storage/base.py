"""
Base classes for storing data
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple, Iterator, Union, Sequence, Any

import numpy as np

from ..grids.base import GridBase
from ..fields import ScalarField, VectorField, Tensor2Field
from ..fields.base import FieldBase
from ..trackers.base import TrackerBase, InfoDict
from ..trackers.intervals import IntervalType, IntervalData
from ..tools.docstrings import fill_in_docstring


    
class StorageBase(metaclass=ABCMeta):
    """ base class for storage of discretized fields """
    
    
    times: Sequence[float]  # :class:`numpy.ndarray`): stored time points
    data: Any  # The actual data for all the stored times
    

    def __init__(self, info: InfoDict = None,
                 write_mode: str = 'truncate_once'):
        """
        Args:
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing one.
                Possible values are: 'append' (data is always appended),
                'truncate' (data is cleared every time this storage is used
                for writing), or 'truncate_once' (data is cleared for the first
                writing, but subsequent data using the same instances are
                appended). Alternatively, specifying 'readonly' will disable
                writing completely.
        """
        self.info = {} if info is None else info
        self.write_mode = write_mode
        self._data_shape: Optional[Tuple[int, ...]] = None
        self._grid: Optional[GridBase] = None
        self._field: Optional[FieldBase] = None
        self._logger = logging.getLogger(self.__class__.__name__)
    
    
    @property
    def data_shape(self) -> Tuple[int, ...]:
        """ the current data shape.
        
        Raises:
            RuntimeError: if data_shape was not set
        """
        if self._data_shape is None:
            raise RuntimeError('data_shape was not set')
        else:  # use the else clause to help typing
            return self._data_shape


    @abstractmethod
    def append(self, data: np.ndarray, time: Optional[float] = None) -> None:
        pass
    
    
    def clear(self, clear_data_shape: bool = False) -> None:
        """ truncate the storage by removing all stored data.
        
        Args:
            clear_data_shape (bool): Flag determining whether the data shape is
                also deleted.
        """
        if clear_data_shape:
            self._data_shape = None
    

    def __len__(self):
        """ return the number of stored items, i.e., time steps """
        return len(self.times)
        

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """ the shape of the stored data """
        if self._data_shape:
            return (len(self),) + self._data_shape 
        else:
            return None
    
    
    @property
    def grid(self) -> Optional[GridBase]:
        """ GridBase: the grid associated with this storage
        
        This returns `None` if grid was not stored in `self.info`.
        """
        if self._grid is None:
            if 'field_attributes' in self.info:
                attrs_serialized = self.info['field_attributes']
                attrs = FieldBase.unserialize_attributes(attrs_serialized)
                self._grid = attrs['grid']
            else:
                self._logger.warning('`grid` attribute was not stored')
        return self._grid
    
    
    def get_field(self, index: int) -> FieldBase:
        """ return the field corresponding to index
        
        Load the data associated with a given index, i.e., with time 
        `self.times[index]`.
        
        Args:
            index (int):
                The index of the data to load
                
        Returns:
            :class:`~pde.fields.FieldBase`:
            The field class containing the grid and data
        """
        if self._field is None:
            # we need to determine the field type
            
            if self.grid is None:
                raise RuntimeError('Could not load grid from data. Please set '
                                   'the `_grid` attribute to the grid that has '
                                   'been used for the stored data.')
            
            if 'field_attributes' in self.info:
                # field type was stored in data
                attrs_serialized = self.info['field_attributes']
                attrs = FieldBase.unserialize_attributes(attrs_serialized)
                self._field = FieldBase.from_state(attrs)
                
            else:
                # try to determine field type automatically

                # obtain data shape by removing the first axis (associated with
                # the time series and the last axes (associated with the spatial
                # dimensions). What is left should be the (local) data stored
                # at each grid point for each time step. Note that self.data
                # might be a list of arrays
                local_shape = self.data[index].shape[:-self.grid.num_axes]
                dim = self.grid.dim
                if len(local_shape) == 0:  # rank 0
                    self._field = ScalarField(self.grid)
                elif local_shape == (dim,):  # rank 1
                    self._field = VectorField(self.grid)
                elif local_shape == (dim, dim):  # rank 2
                    self._field = Tensor2Field(self.grid)
                else:
                    raise RuntimeError('`field` attribute was not stored in '
                                       f'file and the data shape {local_shape} '
                                       'could not be interpreted automatically')
                self._logger.warning('`field` attribute was not stored. We '
                                     'guessed that the data is of type '
                                     f'{self._field.__class__.__name__}.')
                    
        # create the field with the data of the given index
        return self._field.copy(data=self.data[index])
        
        
    def __getitem__(self, key: Union[int, slice]) \
            -> Union[FieldBase, List[FieldBase]]:
        """ return field at given index or a list of fields for a slice """
        if isinstance(key, int):
            return self.get_field(key)
        elif isinstance(key, slice):
            return [self.get_field(i)
                    for i in range(*key.indices(len(self)))]
        else:
            raise TypeError('Unknown key type')
    
    
    def __iter__(self) -> Iterator[FieldBase]:
        """ iterate over all stored fields """
        for i in range(len(self)):
            yield self[i]  # type: ignore
    
    
    def items(self) -> Iterator[Tuple[float, FieldBase]]:
        """ iterate over all times and stored fields, returning pairs """
        for i in range(len(self)):
            yield self.times[i], self[i]  # type: ignore

    
    @fill_in_docstring
    def tracker(self, interval: Union[int, float, IntervalType] = 1) \
            -> "StorageTracker":
        """ create object that can be used as a tracker to fill this storage
        
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
            
        Returns:
            :class:`~pde.trackers.trackers.StorageTracker`
            The tracker that fills the current storage
        """
        return StorageTracker(storage=self, interval=interval)
    
    
    def start_writing(self, field: FieldBase, info: InfoDict = None) -> None:
        """ initialize the storage for writing data
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid
                and the data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        if self.write_mode == 'readonly':
            raise RuntimeError('Cannot write data in readonly mode')
        if self._data_shape is None:
            self._data_shape = field.data.shape
        elif self.data_shape != field.data.shape:
            raise ValueError('Data shape incompatible with stored data')

        self._grid = field.grid
        self._field = field.copy()
        self.info['field_attributes'] = field.attributes_serialized 
#         self.info['grid'] = field.grid.state_serialized
            
    
    def end_writing(self) -> None:
        """ finalize the storage after writing """
        pass


    
class StorageTracker(TrackerBase):
    """ Tracker that stores data in special storage classes 
    
    Attributes:
        storage (:class:`~pde.storage.base.StorageBase`):
            The underlying storage class through which the data can be accessed 
    """

    @fill_in_docstring
    def __init__(self, storage, interval: IntervalData = 1):
        """
        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                Storage instance to which the data is written
            interval:
                {ARG_TRACKER_INTERVAL}
        """
        super().__init__(interval=interval)
        self.storage = storage
        
        
    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """ 
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation        
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        result = super().initialize(field, info)
        self.storage.start_writing(field, info)
        return result
        
        
    def handle(self, field: FieldBase, t: float) -> None:
        """ handle data supplied to this tracker
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float): The associated time
        """
        self.storage.append(field.data, time=t)
    
    
    def finalize(self, info: InfoDict = None) -> None:
        """ finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation        
        """
        super().finalize(info)
        self.storage.end_writing()
    
