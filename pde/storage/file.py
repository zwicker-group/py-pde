"""
Defines a class storing data on the file system using the hierarchical data
format (hdf).

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple  # @UnusedImport

import numpy as np

from ..fields.base import FieldBase
from ..tools.misc import ensure_directory_exists, hdf_write_attributes
from .base import InfoDict, StorageBase


class FileStorage(StorageBase):
    """ store discretized fields in a hdf5 file """

    def __init__(
        self,
        filename: str,
        info: InfoDict = None,
        write_mode: str = "truncate_once",
        max_length: Optional[int] = None,
        compression: bool = True,
        keep_opened: bool = True,
    ):
        """
        Args:
            filename (str):
                The path to the hdf5-file where the data is stored
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing data.
                Possible values are: 'append' (data is always appended),
                'truncate' (data is cleared every time this storage is used
                for writing), or 'truncate_once' (data is cleared for the first
                writing, but appended subsequently). Alternatively, specifying
                'readonly' will disable writing completely.
            max_length (int, optional):
                Maximal number of entries that will be stored in the file. This
                can be used to preallocate data, which can lead to smaller
                files, but is also less flexible. Giving `max_length = None`,
                allows for arbitrarily large data, which might lead to larger
                files.
            compression (bool):
                Whether to store the data in compressed form. Automatically
                enabled chunked storage.
            keep_opened (bool):
                Flag indicating whether the file should be kept opened after
                each writing. If `False`, the file will be closed after writing
                a dataset. This keeps the file in a consistent state, but also
                requires more work before data can be written.
        """
        super().__init__(info=info, write_mode=write_mode)
        self.filename = Path(filename)
        self.compression = compression
        self.keep_opened = keep_opened

        self._logger = logging.getLogger(self.__class__.__name__)
        self._file: Any = None
        self._is_writing = False
        self._data_length: int = None  # type: ignore
        self._max_length: Optional[int] = max_length

        if self.filename.is_file() and self.filename.stat().st_size > 0:
            try:
                self._open("reading")
            except (OSError, KeyError):
                self.close()
                self._logger.warning(
                    f"File `{filename}` could not be opened for reading"
                )

    @property
    def _file_state(self) -> str:
        """ str: the state that the file is currently in """
        if self._file is None:
            return "closed"
        elif self._file.mode == "r":
            return "reading"
        elif self._file.mode == "r+":
            return "writing"
        else:
            raise NotImplementedError(f"Do not understand mode `{self._file.mode}")

    def close(self) -> None:
        """ close the currently opened file """
        if self._file is not None:
            self._logger.info(f"Close file `{self.filename}`")
            self._file.close()
            self._file = None
            self._data_length = None  # type: ignore

    def _create_hdf_dataset(
        self, name: str, shape: Tuple[int, ...] = tuple(), dtype=np.double
    ):
        """create a hdf5 dataset with the given name and data_shape

        Args:
            name (str): Identifier of the hdf5 dataset
            shape (tuple): Data shape of the dataset
            dtype: The data type of the dataset
        """
        if self.compression:
            kwargs = {"chunks": (1,) + shape, "compression": "gzip"}
        else:
            kwargs = {}

        if self._max_length:
            shape = (self._max_length,) + shape
            return self._file.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
        else:
            return self._file.create_dataset(
                name,
                shape=(0,) + shape,
                dtype=dtype,
                maxshape=(None,) + shape,
                **kwargs,
            )

    def _open(self, mode: str = "reading", info: InfoDict = None) -> None:
        """open the hdf file in a particular mode

        Args:
            mode (str):
                Determines how the file is opened. Possible values are
                `reading`, `appending`, `writing`, and `closed`.
            info (dict):
                Supplies extra information that is stored in the storage
        """
        import h5py  # lazy loading so it's not a hard dependence

        state = self._file_state

        if mode == "reading":
            # open file for reading
            if state in ["reading", "appending", "writing"]:
                return  # we can read data when file is open for writing

            # close file to open it again for reading or appending
            if self._file:
                self._file.close()
            self._logger.info(f"Open file `{self.filename}` for reading")
            self._file = h5py.File(self.filename, mode="r")
            self._times = self._file["times"]
            self._data = self._file["data"]
            for k, v in self._file.attrs.items():
                self.info[k] = json.loads(v)
            if info:
                self.info.update(info)

            self._data_shape = self.data.shape[1:]
            self._data_length = self.info.get("data_length")  # type: ignore

        elif mode == "appending":
            # open file for writing without deleting data
            if state in ["appending", "writing"]:
                return  # we are already in a mode where we can append data
            if self.keep_opened and self._is_writing:
                raise RuntimeError(
                    "Currently writing data, so mode cannot be switched."
                )
            if self._file:
                self.close()

            # open file for reading or appending
            self._logger.info(f"Open file `{self.filename}` for appending")
            self._file = h5py.File(self.filename, mode="a")

            if "times" in self._file and "data" in self._file:
                # extract data from datasets in the existing file
                self._times = self._file["times"]
                self._data = self._file["data"]

                # extract information
                for k, v in self._file.attrs.items():
                    self.info[k] = json.loads(v)
                self._data_shape = self.data.shape[1:]
                self._data_length = self.info.get("data_length", self.data.shape[0])

            else:
                # create new datasets
                self._times = self._create_hdf_dataset("times")
                self._data = self._create_hdf_dataset("data", self.data_shape)
                self._data_length = 0

            if info:
                self.info.update(info)

        elif mode == "writing":
            # open file for writing data; delete potentially present data
            if self._is_writing:
                raise RuntimeError("Currently writing data, so mode cannot be switched")
            if self._file:
                self.close()
            else:
                ensure_directory_exists(self.filename.parent)
            self._logger.info(f"Open file `{self.filename}` for writing")
            self._file = h5py.File(self.filename, "w")
            self._times = self._create_hdf_dataset("times")
            self._data = self._create_hdf_dataset("data", self.data_shape)
            if info:
                self.info.update(info)
            self._data_length = 0  # start writing from the beginning

        elif mode == "closed":
            self.close()

        else:
            raise NotImplementedError(f"Mode `{mode}` not implemented")

    def __len__(self):
        """ return the number of stored items, i.e., time steps """
        # determine size of data in HDF5 file
        try:
            length = len(self.times)
        except OSError:
            length = 0

        if self._data_length is None:
            return length
        else:
            # size of stored data is smaller since preallocation was used
            return min(length, self._data_length)

    @property
    def times(self):
        """ :class:`numpy.ndarray`: The times at which data is available """
        self._open("reading")
        return self._times

    @property
    def data(self):
        """  :class:`numpy.ndarray`: The actual data for all time """
        self._open("reading")
        return self._data

    def clear(self, clear_data_shape: bool = False):
        """truncate the storage by removing all stored data.

        Args:
            clear_data_shape (bool): Flag determining whether the data shape is
                also deleted.
        """
        if self._is_writing:
            self._logger.info("Truncate data in hdf5 file")
            # remove data from opened file
            if "times" in self._file:
                del self._file["times"]
            self._times = self._create_hdf_dataset("times")

            if "data" in self._file:
                del self._file["data"]
            self._data = self._create_hdf_dataset("data", self.data_shape)
            self._data_length = 0  # start writing from start

        elif self.filename.is_file():
            self._logger.info("Truncate data by removing hdf5 file")
            self.filename.unlink()

        else:
            self._logger.debug("Truncate is no-op since file does not exist")

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
        if self._is_writing:
            raise RuntimeError(f"{self.__class__.__name__} is already in writing mode")

        # delete data if truncation is requested. This is for instance necessary
        # to remove older data with incompatible data_shape
        if self.write_mode == "truncate" or self.write_mode == "truncate_once":
            self.clear(clear_data_shape=True)

        # initialize the writing, setting current data shape
        super().start_writing(field, info=info)

        # initialize the file for writing with the correct mode
        self._logger.debug(f"Start writing with mode `{self.write_mode}`")
        if self.write_mode == "truncate_once":
            self._open("writing", info)
            self.write_mode = "append"  # do not truncate for next writing

        elif self.write_mode == "truncate":
            self._open("writing", info)

        elif self.write_mode == "append":
            self._open("appending", info)

        elif self.write_mode == "readonly":
            raise RuntimeError("Cannot write in read-only mode")

        else:
            raise ValueError(
                f"Unknown write mode `{self.write_mode}`. Possible values are "
                "`truncate_once`, `truncate`, and `append`"
            )

        if not self.keep_opened:
            # store extra information as attributes
            hdf_write_attributes(self._file, self.info)

        self._is_writing = True

    def _append_data(self, data: np.ndarray, time: float) -> None:
        """append a new data set

        Args:
            data (:class:`numpy.ndarray`): The actual data
            time (float): The time point associated with the data
        """
        if self.keep_opened:
            if not self._is_writing or self._data_length is None:
                raise RuntimeError(
                    "Writing not initialized. Call "
                    f"`{self.__class__.__name__}.start_writing`"
                )

        else:
            # need to reopen the file
            self._open("appending")

        # write the new data
        if self._data_length >= len(self._data):
            self._data.resize((self._data_length + 1,) + self.data_shape)
        self._data[self._data_length] = data

        # write the new time
        if time is None:
            if len(self._times) == 0:
                time = 0
            else:
                time = self._times[self._data_length - 1] + 1
        if self._data_length >= len(self._times):
            self._times.resize((self._data_length + 1,))
        self._times[self._data_length] = time

        self._data_length += 1
        self.info["data_length"] = self._data_length

        if not self.keep_opened:
            self.close()

    def end_writing(self) -> None:
        """finalize the storage after writing.

        This makes sure the data is actually written to a file when
        self.keep_opened == False
        """
        if not self._is_writing:
            return  # writing mode was already ended
        self._logger.debug("End writing")

        # store extra information as attributes
        hdf_write_attributes(self._file, self.info)
        self._file.flush()
        self.close()
        self._is_writing = False
