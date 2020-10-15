"""
Defines a class storing data in memory. 
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from contextlib import contextmanager
from typing import List, Optional, Sequence

import numpy as np

from ..fields import FieldCollection
from ..fields.base import FieldBase
from .base import InfoDict, StorageBase


class MemoryStorage(StorageBase):
    """ store discretized fields in memory """

    def __init__(
        self,
        times: Optional[Sequence[float]] = None,
        data: Optional[List[np.ndarray]] = None,
        field_obj: FieldBase = None,
        info: InfoDict = None,
        write_mode: str = "truncate_once",
    ):
        """
        Args:
            times (:class:`numpy.ndarray`):
                Sequence of times for which data is known
            data (list of :class:`~numpy.ndarray`):
                The field data at the given times
            field_obj (:class:`~pde.fields.base.FieldBase`):
                An instance of the field class that can store data for a single
                time point.
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
        self.times: List[float] = [] if times is None else list(times)
        if field_obj is not None:
            self._field = field_obj.copy()
            self._grid = field_obj.grid
            self._data_shape = field_obj.data.shape

        self.data: List[np.ndarray] = [] if data is None else data
        if self._data_shape is None and len(self.data) > 0:
            self._data_shape = self.data[0].shape

        # check consistency
        if len(self.times) != len(self.data):
            raise ValueError(
                "Length of the supplied `times` and `fields` are inconsistent "
                f"({len(self.times)} != {len(self.data)})"
            )

    @classmethod
    def from_fields(
        cls,
        times: Optional[Sequence[float]] = None,
        fields: Optional[Sequence[FieldBase]] = None,
        info: InfoDict = None,
        write_mode: str = "truncate_once",
    ) -> "MemoryStorage":
        """create MemoryStorage from a list of fields

        Args:
            times (:class:`numpy.ndarray`):
                Sequence of times for which data is known
            fields (list of :class:`~pde.fields.FieldBase`):
                The fields at all given time points
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
        if fields is None:
            field_obj = None
            data = None
        else:
            field_obj = fields[0]
            data = [fields[0].data]
            for field in fields[1:]:
                if field_obj.grid != field.grid:
                    raise ValueError("Grids of the fields are incompatible")
                data.append(field.data)

        return cls(
            times, data=data, field_obj=field_obj, info=info, write_mode=write_mode
        )

    @classmethod
    def from_collection(
        cls, storages: Sequence["StorageBase"], label: str = None
    ) -> "MemoryStorage":
        """combine multiple memory storages into one

        This method can be used to combine multiple time series of different
        fields into a single representation. This requires that all time series
        contain data at the same time points.

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
                raise ValueError("Storages have incompatible times")
            for i, field in enumerate(storage):
                data[i].append(field)

        # convert data format to FieldCollections
        fields = [FieldCollection(d, label=label) for d in data]  # type: ignore

        return cls.from_fields(times, fields=fields)

    def clear(self, clear_data_shape: bool = False) -> None:
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool): Flag determining whether the data shape is
                also deleted.
        """
        self.times = []
        self.data = []
        super().clear(clear_data_shape=clear_data_shape)

    def start_writing(self, field: FieldBase, info: InfoDict = None) -> None:
        """initialize the storage for writing data

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
        if self.write_mode == "truncate_once":
            self.clear()
            self.write_mode = "append"  # do not truncate in subsequent calls

        elif self.write_mode == "truncate":
            self.clear()

        elif self.write_mode == "readonly":
            raise RuntimeError("Cannot write in read-only mode")

        elif self.write_mode != "append":
            raise ValueError(
                f"Unknown write mode `{self.write_mode}`. Possible values are "
                "`truncate_once`, `truncate`, and `append`"
            )

    def _append_data(self, data: np.ndarray, time: float) -> None:
        """append a new data set

        Args:
            data (:class:`numpy.ndarray`): The actual data
            time (float, optional): The time point associated with the data
        """
        assert data.shape == self.data_shape
        self.data.append(np.array(data))  # store copy of the data
        self.times.append(time)


@contextmanager
def get_memory_storage(field: FieldBase, info: InfoDict = None):
    """a context manager that can be used to create a MemoryStorage

    Example:
        This can be used to quickly store data::

            with get_memory_storage(field_class) as storage:
                storage.append(numpy_array0, 0)
                storage.append(numpy_array1, 1)

            # use storage thereafter

    Args:
        field (:class:`~pde.fields.FieldBase`):
            An example of the data that will be written to extract the grid
            and the data_shape
        info (dict):
            Supplies extra information that is stored in the storage

    Yields:
        :class:`MemoryStorage`
    """
    storage = MemoryStorage()
    storage.start_writing(field, info)
    yield storage
    storage.end_writing()
