"""Defines a class storing data using :mod:`modelrunner`.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import modelrunner as mr
import numpy as np

from ..fields.base import FieldBase
from .base import InfoDict, StorageBase, WriteModeType


class ModelrunnerStorage(StorageBase):
    """Store discretized fields in a :mod:`modelrunner` storage.

    This storage class acts as a wrapper for the :mod:`~modelrunner.storage.trajectory`
    module, which allows handling time-dependent data in :mod:`modelrunner` storages.
    In principle, all backends are supported, but it is advisable to use binary formats
    like :class:`~modelrunner.storage.backend.hdf.HDFStorage` or
    :class:`~modelrunner.storage.backend.zarr.ZarrStorage` to write large amounts of
    data.

    .. code-block:: python

        from modelrunner import Result

        r = Result.from_file("data.hdf5")
        r.result.plot()  # plots the final state
        r.storage["trajectory"]  # allows accessing the stored trajectory
    """

    def __init__(
        self,
        storage: mr.storage.StorageGroup,
        *,
        loc: mr.storage.Location = "trajectory",
        info: InfoDict | None = None,
        write_mode: WriteModeType = "truncate_once",
    ):
        """
        Args:
            storage (:class:`~modelrunner.storage.group.StorageGroup`):
                Modelrunner storage used for storing the trajectory
            loc (str or list of str):
                The location in the storage where the trajectory data is written.
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing data. Possible
                values are: 'append' (data is always appended), 'truncate' (data is
                cleared every time this storage is used for writing), or 'truncate_once'
                (data is cleared for the first writing, but appended subsequently).
                Alternatively, specifying 'readonly' will disable writing completely.
        """
        super().__init__(info=info, write_mode=write_mode)
        self.storage = storage
        self.loc = loc
        self._writer: mr.storage.TrajectoryWriter | None = None
        self._reader: mr.storage.Trajectory | None = None

    def close(self) -> None:
        """Close the currently opened trajectory writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> ModelrunnerStorage:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __len__(self):
        """Return the number of stored items, i.e., time steps."""
        return len(self.times)

    @property
    def _io(self) -> mr.storage.TrajectoryWriter | mr.storage.Trajectory:
        """:class:`~modelrunner.storage.group.StorageGroup`: Group with all data."""
        if self._writer is not None:
            return self._writer
        if self._reader is None:
            self._reader = mr.storage.Trajectory(self.storage, loc=self.loc)
        return self._reader

    @property
    def times(self):
        """:class:`~numpy.ndarray`: The times at which data is available."""
        return self._io.times

    @property
    def data(self):
        """:class:`~numpy.ndarray`: The actual data for all time."""
        return self._io._storage.read_array(self._io._loc + ["data"])

    def clear(self, clear_data_shape: bool = False):
        """Truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool):
                Flag determining whether the data shape is also deleted.
        """
        if self.loc in self.storage:
            raise NotImplementedError("Cannot delete existing trajectory")
        super().clear(clear_data_shape=clear_data_shape)

    def start_writing(self, field: FieldBase, info: InfoDict | None = None) -> None:
        """Initialize the storage for writing data.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid and the
                data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        if self._writer:
            raise RuntimeError(f"{self.__class__.__name__} is already in writing mode")
        if self._reader:
            self._reader.close()

        # delete data if truncation is requested. This is for instance necessary
        # to remove older data with incompatible data_shape
        if self.write_mode == "truncate" or self.write_mode == "truncate_once":
            self.clear(clear_data_shape=True)

        # initialize the writing, setting current data shape
        super().start_writing(field, info=info)

        # initialize the file for writing with the correct mode
        self._logger.debug("Start writing with mode '%s'", self.write_mode)
        if self.write_mode == "truncate_once":
            self.write_mode = "append"  # do not truncate for next writing
        elif self.write_mode == "readonly":
            raise RuntimeError("Cannot write in read-only mode")
        elif self.write_mode not in {"truncate", "append"}:
            raise ValueError(
                f"Unknown write mode `{self.write_mode}`. Possible values are "
                "`truncate_once`, `truncate`, and `append`"
            )

        if info:
            self.info.update(info)
        self._writer = mr.storage.TrajectoryWriter(
            self.storage, loc=self.loc, attrs=self.info, mode="append"
        )

    def _append_data(self, data: np.ndarray, time: float) -> None:
        """Append a new data set.

        Args:
            data (:class:`~numpy.ndarray`): The actual data
            time (float): The time point associated with the data
        """
        assert self._writer is not None
        self._writer.append(data, float(time))

    def end_writing(self) -> None:
        """Finalize the storage after writing.

        This makes sure the data is actually written to a file when self.keep_opened ==
        False
        """
        self.close()
