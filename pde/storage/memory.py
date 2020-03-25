"""
Defines a class storing data in memory. 
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Optional, List, Sequence
from contextlib import contextmanager

import numpy as np

from .base import StorageBase, InfoDict
from ..fields import FieldCollection
from ..fields.base import FieldBase 

    
    
class MemoryStorage(StorageBase):
    """ simple storage in memory """
    
    
    def __init__(self, times: Optional[List[float]] = None,
                 fields: Optional[Sequence[FieldBase]] = None,
                 info: InfoDict = None,
                 write_mode: str = 'truncate_once'):
        """
        Args:
            times (:class:`numpy.ndarray`):
                Sequence of times for which data is known
            fields (list of :class:`~pde.fields.FieldBase`):
                The field data at the given times
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing data.
                Possible values are: 'append' (data is always appended),
                'truncate' (data is cleared every time this storage is used
                for writing), or 'truncate_once' (data is cleared for the first
                writing, but appended subsequently). Alternatively, specifying
                'readonly' will disable writing completely.
        """
        super().__init__(info=info, write_mode=write_mode)
        self.times: List[float] = [] if times is None else times
        
        if fields is None:
            self._grid = None
            self.data = []
        else:
            self._grid = fields[0].grid
            self.data = [fields[0].data]
            for field in fields[1:]:
                if self._grid != field.grid:
                    raise ValueError('Grids of the fields are incompatible')
                self.data.append(field.data)
            self._data_shape = self.data[0].shape
            
        # check consistency
        if len(self.times) != len(self.data):
            raise ValueError('Length of the supplied `times` and `fields` are '
                             f'inconsistent ({len(self.times)} != '
                             f'{len(self.data)}).')
            

    @classmethod            
    def from_collection(cls, storages: Sequence["StorageBase"],
                        label: str = None) -> "MemoryStorage":
        """ combine multiple memory storages into one
        
        Args:
            storages (list): 
                A collection of instances of
                :class:`~pde.storage.base.StorageBase` whose data
                will be concatenated into a single MemoryStorage
            label (str, optional):
                The label of the instances of
                :class:`~pde.fields.FieldCollection` that
                represent the concatenated data
                
        Returns:
            :class:`~pde.storage.memory.MemoryStorage`: Storage
            containing all the data.
        """
        if len(storages) == 0:
            return cls()
        
        # initialize the combined data
        times = storages[0].times
        data = [[field] for field in storages[0]]
        
        # append data from further storages
        for storage in storages[1:]:
            if not np.allclose(times, storage.times):
                raise ValueError('Storages have incompatible times')
            for i, field in enumerate(storage):
                data[i].append(field)
                
        # convert data format to FieldCollections
        fields = [FieldCollection(d, label=label)  # type: ignore
                  for d in data]
        
        return cls(times, fields=fields)  # type: ignore
            
        
    def clear(self, clear_data_shape: bool = False) -> None:
        """ truncate the storage by removing all stored data.
        
        Args:
            clear_data_shape (bool): Flag determining whether the data shape is
                also deleted.
        """
        self.times = []
        self.data = []
        super().clear(clear_data_shape=clear_data_shape)


    def start_writing(self, field: FieldBase, info: InfoDict = None) -> None:
        """ initialize the storage for writing data
        
        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid
                and the data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        super().start_writing(field, info=info)
        
        # update info after opening file because otherwise information can be
        # overwritten by data that is already present in the file
        if info is not None:
            self.info.update(info)
            
        # handle the different write modes
        if self.write_mode == 'truncate_once':
            self.clear()
            self.write_mode = 'append'  # do not truncate in subsequent calls
            
        elif self.write_mode == 'truncate':
            self.clear()

        elif self.write_mode == 'readonly':
            raise RuntimeError('Cannot write in read-only mode')
        
        elif self.write_mode != 'append':        
            raise ValueError(f'Unknown write mode `{self.write_mode}`. '
                             'Possible values are `truncate_once`, '
                             '`truncate`, and `append`')
                    
            
    def append(self, data: np.ndarray, time: Optional[float] = None) -> None:
        """ append a new data set
        
        Args:
            data (:class:`numpy.ndarray`): The actual data
            time (float, optional): The time point associated with the data
        """
        assert data.shape == self.data_shape
        self.data.append(np.array(data))  # store copy of the data
        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1
        self.times.append(time)



@contextmanager
def get_memory_storage(field: FieldBase, info: InfoDict = None):
    """ initialize the storage for writing data
    
    Args:
        field (:class:`~pde.fields.FieldBase`):
            An example of the data that will be written to extract the grid
            and the data_shape
        info (dict):
            Supplies extra information that is stored in the storage
    """
    storage = MemoryStorage()
    storage.start_writing(field, info)
    yield storage
    storage.end_writing()
