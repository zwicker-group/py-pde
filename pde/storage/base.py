"""
Base classes for storing data
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import DTypeLike

from ..fields import FieldCollection, ScalarField, Tensor2Field, VectorField
from ..fields.base import FieldBase
from ..grids.base import GridBase
from ..tools.docstrings import fill_in_docstring
from ..tools.output import display_progress
from ..trackers.base import InfoDict, TrackerBase
from ..trackers.interrupts import InterruptsBase, IntervalData

if TYPE_CHECKING:
    from .memory import MemoryStorage  # @UnusedImport


class StorageBase(metaclass=ABCMeta):
    """base class for storing time series of discretized fields

    These classes store time series of :class:`~pde.fields.base.FieldBase`, i.e., they
    store the values of the fields at particular time points. Iterating of the storage
    will return the fields in order and individual time points can also be accessed.
    """

    times: Sequence[float]  # :class:`~numpy.ndarray`): stored time points
    data: Any  # actual data for all the stored times
    write_mode: str  # mode determining how the storage behaves

    def __init__(self, info: InfoDict = None, write_mode: str = "truncate_once"):
        """
        Args:
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing one. Possible
                values are: 'append' (data is always appended), 'truncate' (data is
                cleared every time this storage is used for writing), or 'truncate_once'
                (data is cleared for the first writing, but subsequent data using the
                same instances are appended). Alternatively, specifying 'readonly' will
                disable writing completely.
        """
        self.info = {} if info is None else info
        self.write_mode = write_mode
        self._data_shape: Optional[Tuple[int, ...]] = None
        self._dtype: Optional[DTypeLike] = None
        self._grid: Optional[GridBase] = None
        self._field: Optional[FieldBase] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def data_shape(self) -> Tuple[int, ...]:
        """the current data shape.

        Raises:
            RuntimeError: if data_shape was not set
        """
        if self._data_shape is None:
            raise RuntimeError("data_shape was not set")
        else:  # use the else clause to help typing
            return self._data_shape

    @property
    def dtype(self) -> DTypeLike:
        """the current data type.

        Raises:
            RuntimeError: if data_type was not set
        """
        if self._dtype is None:
            raise RuntimeError("dtype was not set")
        else:  # use the else clause to help typing
            return self._dtype

    @abstractmethod
    def _append_data(self, data: np.ndarray, time: float) -> None:
        pass

    def append(self, field: FieldBase, time: Optional[float] = None) -> None:
        """add field to the storage

        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                The field that is added to the storage
            time (float, optional):
                The time point
        """
        if time is None:
            time = 0 if len(self) == 0 else self.times[-1] + 1

        if self._grid is None:
            self._grid = field.grid
        elif self._grid != field.grid:
            raise ValueError(f"Grids incompatible ({self._grid} != {field.grid})")
        return self._append_data(field.data, time)

    def clear(self, clear_data_shape: bool = False) -> None:
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool):
                Flag determining whether the data shape is also deleted.
        """
        if clear_data_shape:
            self._data_shape = None
            self._dtype = None

    def __len__(self):
        """return the number of stored items, i.e., time steps"""
        return len(self.times)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """the shape of the stored data"""
        if self._data_shape:
            return (len(self),) + self._data_shape
        else:
            return None

    @property
    def has_collection(self) -> bool:
        """bool: whether the storage is storing a collection"""
        if self._field is not None:
            return isinstance(self._field, FieldCollection)
        elif len(self) > 0:
            return isinstance(self._get_field(0), FieldCollection)
        else:
            raise RuntimeError("Storage is empty")

    @property
    def grid(self) -> Optional[GridBase]:
        """GridBase: the grid associated with this storage

        This returns `None` if grid was not stored in `self.info`.
        """
        if self._grid is None:
            if "field_attributes" in self.info:
                attrs_serialized = self.info["field_attributes"]
                attrs = FieldBase.unserialize_attributes(attrs_serialized)

                # load the grid if possible
                if "grid" in attrs:
                    # load grid from only stored field
                    self._grid = attrs["grid"]
                elif "fields" in attrs:
                    # load grid from first field of a collection
                    self._grid = attrs["fields"][0]["grid"]
                else:
                    self._logger.warning(
                        "`grid` attribute was not stored. Available attributes: "
                        + ", ".join(sorted(attrs.keys()))
                    )

            else:
                self._logger.warning("Field attributes are unavailable in info")
        return self._grid

    def _init_field(self) -> None:
        """initialize internal field variable"""
        if self.grid is None:
            raise RuntimeError(
                "Could not load grid from data. Please set the `_grid` attribute "
                "to the grid that has been used for the stored data"
            )

        if "field_attributes" in self.info:
            # field type was stored in data
            attrs_serialized = self.info["field_attributes"]
            attrs = FieldBase.unserialize_attributes(attrs_serialized)
            self._field = FieldBase.from_state(attrs)

        else:
            # try to determine field type automatically

            # obtain data shape by removing the first axis (associated with the time
            # series and the last axes (associated with the spatial dimensions). What is
            # left should be the (local) data stored at each grid point for each time
            # step. Note that self.data might be a list of arrays
            local_shape = self.data[0].shape[: -self.grid.num_axes]
            dim = self.grid.dim
            if len(local_shape) == 0:  # rank 0
                self._field = ScalarField(self.grid, dtype=self.data[0].dtype)
            elif local_shape == (dim,):  # rank 1
                self._field = VectorField(self.grid, dtype=self.data[0].dtype)
            elif local_shape == (dim, dim):  # rank 2
                self._field = Tensor2Field(self.grid, dtype=self.data[0].dtype)
            else:
                raise RuntimeError(
                    "`field` attribute was not stored in file and the data shape "
                    f"{local_shape} could not be interpreted automatically"
                )
            self._logger.warning(
                "`field` attribute was not stored. We guessed that the data is of "
                f"type {self._field.__class__.__name__}."
            )

    def _get_field(self, t_index: int) -> FieldBase:
        """return the field corresponding to the given time index

        Load the data given an index, i.e., the data at time `self.times[t_index]`.

        Args:
            t_index (int):
                The index of the data to load

        Returns:
            :class:`~pde.fields.FieldBase`:
            The field class containing the grid and data
        """
        if t_index < 0:
            t_index += len(self)

        if not 0 <= t_index < len(self):
            raise IndexError("Time index out of range")

        if self._field is None:
            self._init_field()

        # create the field with the data of the given index
        assert self._field is not None
        field = self._field.copy()
        field.data = self.data[t_index]
        return field

    def __getitem__(self, key: Union[int, slice]) -> Union[FieldBase, List[FieldBase]]:
        """return field at given index or a list of fields for a slice"""
        if isinstance(key, int):
            return self._get_field(key)
        elif isinstance(key, slice):
            return [self._get_field(i) for i in range(*key.indices(len(self)))]
        else:
            raise TypeError("Unknown key type")

    def __iter__(self) -> Iterator[FieldBase]:
        """iterate over all stored fields"""
        for i in range(len(self)):
            yield self[i]  # type: ignore

    def items(self) -> Iterator[Tuple[float, FieldBase]]:
        """iterate over all times and stored fields, returning pairs"""
        for i in range(len(self)):
            yield self.times[i], self[i]  # type: ignore

    @fill_in_docstring
    def tracker(
        self, interval: Union[int, float, InterruptsBase] = 1
    ) -> "StorageTracker":
        """create object that can be used as a tracker to fill this storage

        Args:
            interval:
                {ARG_TRACKER_INTERVAL}

        Returns:
            :class:`StorageTracker`: The tracker that fills the current storage
        """
        return StorageTracker(storage=self, interval=interval)

    def start_writing(self, field: FieldBase, info: InfoDict = None) -> None:
        """initialize the storage for writing data

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid
                and the data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        if self.write_mode == "readonly":
            raise RuntimeError("Cannot write data in readonly mode")

        if self._data_shape is None:
            self._data_shape = field.data.shape
        elif self.data_shape != field.data.shape:
            raise ValueError("Data shape incompatible with stored data")

        if self._dtype is None:
            self._dtype = field.dtype

        self._grid = field.grid
        self._field = field.copy()
        self.info["field_attributes"] = field.attributes_serialized

    def end_writing(self) -> None:
        """finalize the storage after writing"""
        pass

    def extract_field(
        self, field_id: Union[int, str], label: str = None
    ) -> "MemoryStorage":
        """extract the time course of a single field from a collection

        Note:
            This might return a view into the original data, so modifying the returned
            data can also change the underlying original data.

        Args:
            field_id (int or str):
                The index into the field collection. This determines which field of the
                collection is returned. Instead of a numerical index, the field label
                can also be supplied. If there are multiple fields with the same label,
                only the first field is returned.
            label (str):
                The label of the returned field. If omitted, the stored label is used.

        Returns:
            :class:`MemoryStorage`: a storage instance that contains the data for the
            single field
        """
        from .memory import MemoryStorage  # @Reimport

        if self._field is None:
            self._init_field()

        # get the field to check its type
        if not isinstance(self._field, FieldCollection):
            raise TypeError(
                "Can only extract fields from `FieldCollection`. Current storage "
                f"stores `{self._field.__class__.__name__}`."
            )

        # determine the field index
        if isinstance(field_id, str):
            field_index = self._field.labels.index(field_id)
        else:
            field_index = field_id

        # extract the field and the associated time series
        field_obj = self._field[field_index].copy()
        if label:
            field_obj.label = label
        field_slice = self._field._slices[field_index]
        data = [d[field_slice].reshape(field_obj.data.shape) for d in self.data]

        # create the corresponding MemoryStorage
        return MemoryStorage(
            times=self.times, data=data, field_obj=field_obj, info=self.info
        )

    def extract_time_range(
        self, t_range: Union[float, Tuple[float, float]] = None
    ) -> "MemoryStorage":
        """extract a particular time interval

        Note:
            This might return a view into the original data, so modifying the returned
            data can also change the underlying original data.

        Args:
            t_range (float or tuple):
                Determines the range of time points included in the result. If only a
                single number is given, all data up to this time point are included.

        Returns:
            :class:`MemoryStorage`: a storage instance that contains the extracted data.
        """
        from .memory import MemoryStorage  # @Reimport

        # get the time bracket
        try:
            t_start, t_end = t_range  # type: ignore
        except TypeError:
            t_start, t_end = None, t_range
        if t_start is None:
            t_start = self.times[0]
        if t_end is None:
            t_end = self.times[-1]

        # determine the associated indices
        i_start = np.searchsorted(self.times, t_start, side="left")
        i_end = np.searchsorted(self.times, t_end, side="right")

        # extract the actual memory
        return MemoryStorage(
            times=self.times[i_start:i_end],  # type: ignore
            data=self.data[i_start:i_end],  # type: ignore
            field_obj=self._field,
            info=self.info,
        )

    def apply(
        self, func: Callable, out: StorageBase = None, *, progress: bool = False
    ) -> StorageBase:
        """applies function to each field in a storage

        Args:
            func (callable):
                The function to apply to each stored field. The function must either
                take as a single argument the field or as two arguments the field and
                the associated time point. In both cases, it should return a field.
            out (:class:`~pde.storage.base.StorageBase`):
                Storage to which the output is written. If omitted, a new
                :class:`~pde.storage.memory.MemoryStorage` is used and returned
            progress (bool):
                Flag indicating whether the progress is shown during the calculation

        Returns:
            :class:`~pde.storage.base.StorageBase`: The new storage that contains the
            data after the function `func` has been applied
        """
        # get the number of arguments that the user function expects
        num_args = len(signature(func).parameters)
        writing = False  # flag indicating whether output storage was opened

        for t, field in display_progress(
            self.items(), total=len(self), enabled=progress
        ):
            # apply the user function
            if num_args == 0:
                transformed = func()
            elif num_args == 1:
                transformed = func(field)
            else:
                transformed = func(field, t)

            if not isinstance(transformed, FieldBase):
                raise TypeError("The user function must return a field")

            if out is None:
                from .memory import MemoryStorage  # @Reimport

                out = MemoryStorage(field_obj=transformed)

            if not writing:
                out.start_writing(transformed)
                writing = True

            out.append(transformed, t)

        if writing:
            out.end_writing()  # type: ignore

        # make sure that a storage is returned, even when no fields are present
        if out is None:
            from .memory import MemoryStorage  # @Reimport

            out = MemoryStorage()

        return out

    def copy(self, out: StorageBase = None, *, progress: bool = False) -> StorageBase:
        """copies all fields in a storage to a new one

        Args:
            out (:class:`~pde.storage.base.StorageBase`):
                Storage to which the output is written. If omitted, a new
                :class:`~pde.storage.memory.MemoryStorage` is used and returned
            progress (bool):
                Flag indicating whether the progress is shown during the calculation

        Returns:
            :class:`~pde.storage.base.StorageBase`: The new storage that contains the
            copied data
        """
        # apply the identity function to do the copy
        return self.apply(lambda x: x, out=out, progress=progress)


class StorageTracker(TrackerBase):
    """Tracker that stores data in special storage classes

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
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float): The associated time
        """
        self.storage.append(field, time=t)

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        self.storage.end_writing()
