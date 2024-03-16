"""
Defines a class storing data on the file system as a compressed movie

This package requires the optional :mod:`ffmpeg-python` package to use FFmpeg for
reading and writing movies.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike

from ..fields import FieldCollection, ScalarField
from ..fields.base import FieldBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from ..trackers.interrupts import ConstantInterrupts
from . import _ffmpeg as FFmpeg
from .base import InfoDict, StorageBase, StorageTracker, WriteModeType


def _get_limits(value: float | ArrayLike, dim: int) -> np.ndarray:
    """helper function creating sequence of length `dim` from input"""
    if np.isscalar(value):
        return np.full(dim, value)
    else:
        return np.atleast_1d(value)[:dim]


class MovieStorage(StorageBase):

    codecs: list[str] = ["libx264", "libx264rgb", "libx265"]
    """list: List of prefered codecs"""

    def __init__(
        self,
        filename: str | Path,
        *,
        vmin: float | ArrayLike = 0,
        vmax: float | ArrayLike = 1,
        bitrate: int = -1,
        info: InfoDict | None = None,
        write_mode: WriteModeType = "truncate_once",
    ):
        """
        Args:
            filename (str):
                The path to the hdf5-file where the data is stored
            vmin (float or array):
                Lowest values that are encoded (per field)
            vmax (float or array):
                Highest values that are encoded (per field)
            bitrate (float):
                The bitrate of the movie (in kilobits per second). The default value of
                -1 let's the encode choose an appropriate bit rate.
            info (dict):
                Supplies extra information that is stored in the storage
            write_mode (str):
                Determines how new data is added to already existing data. Possible
                values are: 'append' (data is always appended), 'truncate' (data is
                cleared every time this storage is used for writing), or 'truncate_once'
                (data is cleared for the first writing, but appended subsequently).
                Alternatively, specifying 'readonly' will disable writing completely.


        TODO:
            - allow more bits for colorchannels
            - allow choosing bitrate for video
            - support different write_mode
            - support different video codecs (from a list of priorities)
               Check in order which one supports the chosen pixel format (and store that
               information)
            - track whether times roughly work (checking for frame drops)
            - we could embedd extra information (like time, and maybe colorscaling) in
              the individual frames if we extended the shape
            - introduce encode/decode method for images
              (provide abstract classes that manage bit depth, color channels and ffmpeg
              codes)
        """
        if not module_available("ffmpeg"):
            raise ModuleNotFoundError("`MovieStorage` needs `ffmpeg-python` package")

        super().__init__(info=info, write_mode=write_mode)
        self.filename = Path(filename)
        self.vmin = vmin
        self.vmax = vmax
        self.bitrate = bitrate

        self._logger = logging.getLogger(self.__class__.__name__)
        self._ffmpeg: Any = None
        self._state: Literal["closed", "reading", "writing"] = "closed"
        self._norms: list[Normalize] | None = None
        self._is_writing = False

    def __del__(self):
        self.close()  # ensure open files are closed when the FileStorage is deleted

    def close(self) -> None:
        """close the currently opened file"""
        if self._ffmpeg is not None:
            self._logger.info(f"Close movie file `{self.filename}`")
            if self._state == "writing":
                self._ffmpeg.stdin.close()
                self._ffmpeg.wait()
            elif self._state == "reading":
                self._ffmpeg.stdout.close()
                self._ffmpeg.wait()
            self._ffmpeg = None
            self._state = "closed"

    def __enter__(self) -> MovieStorage:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def clear(self):
        """truncate the storage by removing all stored data."""
        if self.filename.exists():
            self.filename.unlink()

    def _get_metadata(self) -> str:
        """obtain metadata stored in the video"""
        info = self.info.copy()
        info["version"] = 1
        info["vmin"] = self.vmin
        info["vmax"] = self.vmax
        return json.dumps(info)

    def _read_metadata(self) -> None:
        """read metadata from video and store it in :attr:`info`"""
        import ffmpeg

        info = ffmpeg.probe(self.filename)
        # TODO: Check stored pixel format against the one

        # sanity checks on the video
        nb_streams = info["format"]["nb_streams"]
        if nb_streams != 1:
            self._logger.warning(f"Only using first of {nb_streams} streams")

        raw_comment = info["format"].get("tags", {}).get("comment", "{}")
        metadata = json.loads(shlex.split(raw_comment)[0])

        version = metadata.pop("version", 1)
        if version != 1:
            self._logger.warning(f"Unknown metadata version `{version}`")
        self.vmin = metadata.pop("vmin", 0)
        self.vmax = metadata.pop("vmax", 1)
        self.info.update(metadata)

        stream = info["streams"][0]
        self.info["num_frames"] = int(stream["nb_frames"])
        self.info["width"] = stream["coded_width"]
        self.info["height"] = stream["coded_height"]
        if self.info["pixel_format"] != stream["pix_fmt"]:
            self._logger.warning(
                f"Inconsistent pixel format {self.info['pixel_format']} != "
                f"{stream['pix_fmt']}"
            )
        if self.info["pixel_format"] not in FFmpeg.pixel_formats:
            self._logger.warning(f"Unknown pixel format {self.info['pixel_format']}")

    def _init_normalization(self, field: FieldBase, *, inverse: bool = False) -> None:
        """initialize the normalizations of the color information

        Args:
            field (:class:`~pde.fields.base.FieldBase):
                Example field to obtain information about grid and data rank
            inverse (bool):
                Whether inverse normalization function should be returned
        """
        self._norms = []
        fields = field.fields if isinstance(field, FieldCollection) else [field]
        vmin = _get_limits(self.vmin, len(fields))
        vmax = _get_limits(self.vmax, len(fields))
        for f_id, f in enumerate(fields):
            if inverse:
                norm = lambda data: vmin[f_id] + (vmax[f_id] - vmin[f_id]) * data
            else:
                norm = Normalize(vmin[f_id], vmax[f_id], clip=True)
            num = self.grid.dim**f.rank  # independent components in the field
            self._norms.extend([norm] * num)

    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        """reshape data such that it has exactly three dimensions

        Assumes that that the input data has the spatial dimension last. The returned
        data will have two spatial dimensions and all other dimensions (tensorial and
        field collections) collapsed to the last dimension
        """
        data = np.copy(data)  # safety, so we don't change the shape of the original

        # make sure there are two spatial dimensions
        grid_dim = self._grid.num_axes
        if grid_dim > 2:
            raise NotImplementedError
        if grid_dim == 1:
            data = data.reshape(data.shape + (1,))
        assert data.ndim >= 2
        if data.ndim == 2:
            return data.reshape(data.shape + (1,))  # explicitly scalar data
        elif data.ndim > 3:
            # collapse first dimensions, so we have three in total
            data = data.reshape((-1,) + data.shape[-2:])
        return np.moveaxis(data, 0, -1)

    def start_writing(self, field: FieldBase, info: InfoDict | None = None) -> None:
        """initialize the storage for writing data

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid and the
                data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        import ffmpeg

        if self._is_writing:
            raise RuntimeError(f"{self.__class__.__name__} is already in writing mode")
        if self._ffmpeg is not None:
            raise RuntimeError("ffmpeg process already started")

        # delete data if truncation is requested. This is for instance necessary
        # to remove older data with incompatible data_shape
        if self.write_mode == "truncate" or self.write_mode == "truncate_once":
            self.clear()

        # initialize the writing, setting current data shape
        super().start_writing(field, info=info)
        if info:
            self.info.update(info)

        # get spatial dimension of the video
        self._grid = field.grid
        if self._grid.num_axes == 1:
            width, height = field.grid.shape[0], 1
        elif self._grid.num_axes == 2:
            width, height = field.grid.shape
        else:
            raise RuntimeError("Cannot use grid with more than two axes")

        # get color channel information
        if isinstance(field, ScalarField):
            codec = "libx264"
            self.info["pixel_format"] = "gray"
        else:
            codec = "libx264rgb"
            self.info["pixel_format"] = "rgb24"
        self._pix_fmt = FFmpeg.pixel_formats[self.info["pixel_format"]]
        self._data_shape = (width, height, self._pix_fmt.channels)
        self.info["frame_shape"] = self._data_shape
        self.info["field_shape"] = field.data.shape
        if field.is_complex:
            raise NotImplementedError("MovieStorage does not support complex values")

        # set up the normalization
        self._init_normalization(field)

        # set input
        self._logger.debug(f"Start ffmpeg process for `{self.filename}`")
        args = {
            "format": "rawvideo",
            "s": f"{width}x{height}",
            "pix_fmt": self._pix_fmt.pix_fmt,
            "loglevel": "warning",
        }
        f_input = ffmpeg.input("pipe:", **args)
        # set output format
        args = {
            "vcodec": codec,
            "crf": "0",  # Constant Rate Factor - aim for lossless compression
            "pix_fmt": self._pix_fmt.pix_fmt,
            "metadata": "comment=" + shlex.quote(self._get_metadata()),
        }
        if self.bitrate > 0:
            args["video_bitrate"] = self.bitrate
        f_output = f_input.output(filename=self.filename, **args)
        if self.write_mode in {"truncate", "truncate_once"}:
            f_output = f_output.overwrite_output()  # allow overwriting file
        self._ffmpeg = f_output.run_async(pipe_stdin=True)  # start process
        self._state = "writing"

    def _append_data(self, data: np.ndarray, time: float) -> None:
        """append a new data set

        Args:
            data (:class:`~numpy.ndarray`): The actual data
            time (float): The time point associated with the data (currently not used)
        """
        if self._state != "writing" or self._ffmpeg is None:
            RuntimeError(
                "Writing not initialized. Call "
                f"`{self.__class__.__name__}.start_writing`"
            )
        # ensure the data has the shape width x height x depth
        data = self._reshape_data(data)
        assert len(self._data_shape) == data.ndim == 3
        assert data.shape[2] == len(self._norms)
        if self._data_shape[2] == 1:
            # single color channel
            data = data.reshape(self._data_shape)
        elif self._data_shape[2] == 3:
            # three color channels
            if data.shape[2] < 3:
                zero_shape = data.shape[:2] + (3 - data.shape[2],)
                data = np.dstack((data, np.zeros(zero_shape, dtype=self._dtype)))
        else:
            raise RuntimeError
        assert data.shape == self._data_shape

        # map values to [0, 1] float values
        for i, norm in enumerate(self._norms):
            data[..., i] = norm(data[..., i])

        # write the new data
        self._ffmpeg.stdin.write(self._pix_fmt.data_to_frame(data).tobytes())

    def end_writing(self) -> None:
        """finalize the storage after writing"""
        if not self._state == "writing":
            return  # writing mode was already ended
        self._logger.debug("End writing")
        self.close()

    def __len__(self):
        """return the number of stored items, i.e., time steps"""
        if "num_frames" not in self.info:
            self._read_metadata()
        return self.info["num_frames"]

    @property
    def times(self):
        """:class:`~numpy.ndarray`: The times at which data is available"""
        t_start = self.info.get("t_start")
        if t_start is None:
            t_start = 0
        dt = self.info.get("dt", 1)
        return t_start + dt * np.arange(len(self))

    @property
    def data(self):
        """:class:`~numpy.ndarray`: The actual data for all time"""
        raise NotImplementedError

    def __iter__(self) -> Iterator[FieldBase]:
        """iterate over all stored fields"""
        import ffmpeg

        if "width" not in self.info:
            self._read_metadata()
        if self._field is None:
            self._init_field()
        self._init_normalization(self._field, inverse=True)
        pix_fmt = FFmpeg.pixel_formats[self.info["pixel_format"]]
        frame_shape = (self.info["width"], self.info["height"], pix_fmt.channels)
        data_shape = (self.info["width"], self.info["height"], len(self._norms))
        data = np.empty(data_shape, dtype=self._dtype)

        # iterate over entire movie
        f_input = ffmpeg.input(self.filename, loglevel="warning")
        f_output = f_input.output(
            "pipe:", format="rawvideo", pix_fmt=pix_fmt.pix_fmt, vframes=8
        )
        proc = f_output.run_async(pipe_stdout=True)
        while True:
            read_bytes = proc.stdout.read(np.prod(data_shape))
            if not read_bytes:
                break
            frame = np.frombuffer(read_bytes, pix_fmt.dtype).reshape(frame_shape)

            for i, norm in enumerate(self._norms):
                data[..., i] = norm(pix_fmt.data_from_frame(frame[:, :, i]))

            # create the field with the data of the given index
            assert self._field is not None
            field = self._field.copy()
            field.data = data.reshape(self.info["field_shape"])
            yield field

    def items(self) -> Iterator[tuple[float, FieldBase]]:
        """iterate over all times and stored fields, returning pairs"""
        # iterate over entire movie
        t = self.info.get("t_start", 0)
        dt = self.info.get("dt", 1)
        for field in self:
            yield t, field
            t += dt

    @fill_in_docstring
    def tracker(
        self,
        interrupts: ConstantInterrupts | float = 1,
        *,
        transformation: Callable[[FieldBase, float], FieldBase] | None = None,
    ) -> StorageTracker:
        """create object that can be used as a tracker to fill this storage

        Args:
            interrupts (:class:`~pde.tracker.ConstantInterrupts` or float):
                Time interval with which the tracker is being called
            transformation (callable, optional):
                A function that transforms the current state into a new field or field
                collection, which is then stored. This allows to store derived
                quantities of the field during calculations. The argument needs to be a
                callable function taking 1 or 2 arguments. The first argument always is
                the current field, while the optional second argument is the associated
                time.

        Returns:
            :class:`StorageTracker`: The tracker that fills the current storage

        Example:
            The `transformation` argument allows storing additional fields:

            .. code-block:: python

                def add_to_state(state):
                    transformed_field = state.smooth(1)
                    return field.append(transformed_field)

                storage = pde.MemoryStorage()
                tracker = storage.tracker(1, transformation=add_to_state)
                eq.solve(..., tracker=tracker)

            In this example, :obj:`storage` will contain a trajectory of the fields of
            the simulation as well as the smoothed fields. Other transformations are
            possible by defining appropriate :func:`add_to_state`
        """
        if np.isscalar(interrupts):
            interrupts = ConstantInterrupts(interrupts)
        if not isinstance(interrupts, ConstantInterrupts):
            self._logger.warning("`VideoTracker` can only use `ConstantInterrupts`")
        self.info["dt"] = interrupts.dt
        self.info["t_start"] = interrupts.t_start
        return StorageTracker(
            storage=self, interrupts=interrupts, transformation=transformation
        )
