"""Defines a class storing data on the file system as a compressed movie.

This package requires the optional :mod:`ffmpeg-python` package to use FFmpeg for
reading and writing movies.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import io
import json
import shlex
import types
import warnings
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike

from ..fields import FieldCollection
from ..fields.base import FieldBase
from ..fields.datafield_base import DataFieldBase
from ..tools import ffmpeg as FFmpeg
from ..tools.cache import cached_property
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from ..tools.parse_duration import parse_duration
from ..trackers.interrupts import ConstantInterrupts, InterruptData, parse_interrupt
from .base import InfoDict, StorageBase, StorageTracker, WriteModeType


def _get_limits(value: float | ArrayLike, dim: int) -> np.ndarray:
    """Helper function creating sequence of length `dim` from input."""
    if np.isscalar(value):
        return np.full(dim, value, dtype=float)
    else:
        return np.asarray(value)[:dim].astype(float)  # type: ignore


def _import_ffmpeg() -> types.ModuleType:
    """Import `ffmpeg` package, warning when incorrect package is installed."""
    # try to figure out which `ffmpeg` package is installed
    try:
        from importlib.metadata import packages_distributions  # type: ignore

    except ImportError:
        pass  # the packages_distributions function was only added in python 3.10

    else:
        # check whether `ffmpeg` refers to the correct package
        packages = packages_distributions().get("ffmpeg", [])
        if len(packages) == 1:
            name = packages[0]
            if name != "ffmpeg-python":
                warnings.warn(
                    f"Expected `ffmpeg-python` package, but found `{name}`",
                    ImportWarning,
                )
        elif len(packages) > 1:
            warnings.warn(
                f"Expected `ffmpeg-python` package, but found {packages}", ImportWarning
            )

    # import package
    import ffmpeg

    return ffmpeg  # type: ignore


class MovieStorage(StorageBase):
    """Store discretized fields in a movie file.

    This storage only works when the `ffmpeg` program and :mod:`ffmpeg` is installed.
    The default codec is `FFV1 <https://en.m.wikipedia.org/wiki/FFV1>`_, which supports
    lossless compression for various configurations. Not all video players support this
    codec, but `VLC <https://www.videolan.org>`_ usually works quite well.

    Note that important meta information is stored as a comment in the movie, so this
    data must not be deleted or altered if the video should be read again.

    Warning:
        This storage potentially compresses data and can thus lead to loss of some
        information. The data quality depends on many parameters, but most important are
        the bits per channel of the video format and the range that is encoded
        (determined by `vmin` and `vmax`).

        Note also that selecting individual time points might be quite slow since the
        video needs to be read from the beginning each time. Instead, it is much more
        efficient to process entire videos (by iterating over them or using
        :func:`~pde.storage.movie.MovieStorage.items()`).
    """

    def __init__(
        self,
        filename: str | Path,
        *,
        vmin: float | ArrayLike = 0,
        vmax: float | ArrayLike = 1,
        bits_per_channel: int = 16,
        video_format: str = "auto",
        bitrate: int = -1,
        info: InfoDict | None = None,
        write_mode: WriteModeType = "truncate_once",
        write_times: bool = False,
        loglevel: str = "warning",
    ):
        """
        Args:
            filename (str):
                The path where the movie is stored. The file extension determines the
                container format of the movie. The standard codec FFV1 plays well with
                the ".avi", ".mkv", and ".mov" container format.
            vmin (float or array):
                Lowest values that are encoded (per field). Smaller values are clipped
                to this value.
            vmax (float or array):
                Highest values that are encoded (per field). Larger values are clipped
                to this value.
            bits_per_channel (int):
                The number of bits used per color channel. Typical values are 8 and 16.
                The relative accuracy of stored values is 0.01 and 0.0001, respectively.
            video_format (str):
                Identifier for a video format from :data:`~pde.tools.ffmpeg.formats`,
                which determines the number of channels, the bit depth of individual
                colors, and the codec. The special value `auto` tries to find a suitable
                format automatically, taking `bits_per_channel` into account.
            bitrate (float):
                The bitrate of the movie (in kilobits per second). The default value of
                -1 let's the encoder choose an appropriate bit rate.
            info (dict):
                Supplies extra information that is stored in the storage alongside
                additional information necessary to reconstruct fields and grids.
            write_mode (str):
                Determines how new data is added to already existing data. Possible
                values are: 'append' (data is always appended), 'truncate' (data is
                cleared every time this storage is used for writing), or 'truncate_once'
                (data is cleared for the first writing, but appended subsequently).
                Alternatively, specifying 'readonly' will disable writing completely.
            write_times (bool):
                Flag determining whether timestamps are written to a file. If True, a
                separate file with name :code:`filename + ".times"` is created where the
                times are written as plain text. Without these timestamps, the time
                information might be inaccurate.
            loglevel (str):
                FFmpeg log level determining the amount of data sent to stdout. The
                default only emits warnings and errors, but setting this to `"info"` can
                be useful to get additional information about the encoding.
        """
        if not module_available("ffmpeg"):
            raise ModuleNotFoundError("`MovieStorage` needs `ffmpeg-python` package")

        super().__init__(info=info, write_mode=write_mode)
        self.filename = Path(filename)
        self.vmin = vmin
        self.vmax = vmax
        self.bits_per_channel = bits_per_channel
        self.video_format = video_format
        self.bitrate = bitrate
        self.loglevel = loglevel
        self.write_times = write_times

        self._ffmpeg: Any = None
        self._times_file: io.TextIOBase | None = None
        self._state: Literal["closed", "reading", "writing"] = "closed"
        self._norms: list[Normalize] | None = None
        self._is_writing = False

    def __del__(self):
        self.close()  # ensure open files are closed when the FileStorage is deleted

    def close(self) -> None:
        """Close the currently opened file."""
        if self._ffmpeg is not None:
            self._logger.info("Close movie file '%s'", self.filename)
            if self._state == "writing":
                self._ffmpeg.stdin.close()
                self._ffmpeg.wait()
            elif self._state == "reading":
                self._ffmpeg.stdout.close()
                self._ffmpeg.wait()
            self._ffmpeg = None

        if self._times_file is not None:
            self._times_file.close()
            self._times_file = None

        self._state = "closed"

    def __enter__(self) -> MovieStorage:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def clear(self):
        """Truncate the storage by removing all stored data."""
        if self.filename.exists():
            self.filename.unlink()

    def _get_metadata(self) -> str:
        """Obtain metadata stored in the video."""
        info = self.info.copy()
        info["version"] = 1
        info["vmin"] = self.vmin
        info["vmax"] = self.vmax
        info["write_times"] = self.write_times
        return json.dumps(info)

    def _read_metadata(self) -> None:
        """Read metadata from video and store it in :attr:`info`"""
        ffmpeg = _import_ffmpeg()  # lazy loading so it's not a hard dependence

        path = Path(self.filename)
        if not path.exists():
            raise OSError(f"File `{path}` does not exist")
        info = ffmpeg.probe(path)

        # sanity checks on the video
        nb_streams = info["format"]["nb_streams"]
        if nb_streams != 1:
            self._logger.warning("Only using first of %d streams", nb_streams)

        tags = info["format"].get("tags", {})
        # read comment field, which can be either lower case or upper case
        raw_comment = tags.get("comment", tags.get("COMMENT", "{}"))
        if raw_comment == "{}":
            self._logger.warning("Could not find metadata written by `py-pde`")
        metadata = json.loads(shlex.split(raw_comment)[0])

        version = metadata.pop("version", 1)
        if version != 1:
            self._logger.warning("Unknown metadata version `%d`", version)
        self.vmin = metadata.pop("vmin", 0)
        self.vmax = metadata.pop("vmax", 1)
        self.info.update(metadata)

        # read information from the first stream
        stream = info["streams"][0]
        try:
            self.info["num_frames"] = int(stream["nb_frames"])
        except KeyError:
            # frame count is not directly in the video
            # => try determining it from the duration
            try:
                fps = Fraction(stream.get("avg_frame_rate", None))
                duration = parse_duration(stream.get("tags", {}).get("DURATION"))
            except TypeError as err:
                raise RuntimeError("Frame count could not be read from video") from err
            else:
                self.info["num_frames"] = int(duration.total_seconds() * float(fps))
        self.info["width"] = stream["width"]
        self.info["height"] = stream["height"]
        if self.video_format == "auto":
            video_format = self.info.get("video_format")
            if video_format is None:
                video_format = stream.get("pix_fmt")
            if video_format is None:
                raise RuntimeError("Could not determine video format from file")
        else:
            video_format = self.video_format
        try:
            self._format = FFmpeg.formats[video_format]
        except KeyError:
            self._logger.warning("Unknown pixel format `%s`", video_format)
        else:
            if self._format.pix_fmt_file != stream.get("pix_fmt"):
                self._logger.info(
                    "Pixel format differs from requested one: %s != %s",
                    self._format.pix_fmt_file,
                    stream.get("pix_fmt"),
                )

    def _init_normalization(self, field: FieldBase) -> None:
        """Initialize the normalizations of the color information.

        Args:
            field (:class:`~pde.fields.base.FieldBase):
                Example field to obtain information about grid and data rank
            inverse (bool):
                Whether inverse normalization function should be returned

        The resulting normalization functions are stored in `self._norms`
        """
        self._norms = []
        if isinstance(field, FieldCollection):
            fields: Sequence[DataFieldBase] = field.fields
        else:
            fields = [field]  # type: ignore
        vmin = _get_limits(self.vmin, len(fields))
        vmax = _get_limits(self.vmax, len(fields))
        for f_id, f in enumerate(fields):
            norm = Normalize(vmin[f_id], vmax[f_id], clip=True)  # type: ignore
            num = f.grid.dim**f.rank  # independent components in the field
            self._norms.extend([norm] * num)

    @property
    def _filename_times(self) -> Path:
        """str: the filename of the file to whichtime stamps are written"""
        return self.filename.with_suffix(self.filename.suffix + ".times")

    def start_writing(self, field: FieldBase, info: InfoDict | None = None) -> None:
        """Initialize the storage for writing data.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be written to extract the grid and the
                data_shape
            info (dict):
                Supplies extra information that is stored in the storage
        """
        ffmpeg = _import_ffmpeg()  # lazy loading so it's not a hard dependence

        if self._is_writing:
            raise RuntimeError(f"{self.__class__.__name__} is already in writing mode")
        if self._ffmpeg is not None:
            raise RuntimeError("ffmpeg process already started")

        # delete data if truncation is requested. This is for instance necessary
        # to remove older data with incompatible data_shape
        if self.write_mode == "truncate":
            self.clear()
        elif self.write_mode == "truncate_once":
            self.clear()
            self.write_mode = "append"  # do not truncate in subsequent calls
        elif self.write_mode == "append":
            raise NotImplementedError("Appending to movies is not possible")

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
        if self.video_format == "auto":
            channels = field._data_flat.shape[0]
            video_format = FFmpeg.find_format(channels, self.bits_per_channel)
            if video_format is None:
                raise RuntimeError(
                    f"Could not find a video format with {channels} channels and "
                    f"{self.bits_per_channel} bits per channel."
                )
            self.info["video_format"] = video_format
        else:
            self.info["video_format"] = self.video_format
        self._format = FFmpeg.formats[self.info["video_format"]]
        if field.is_complex:
            raise NotImplementedError("MovieStorage does not support complex values")
        self._frame_shape = (width, height, self._format.channels)

        # set up the normalization
        self._init_normalization(field)

        # set input
        self._logger.debug("Start ffmpeg process for `%s`", self.filename)
        input_args = {
            "format": "rawvideo",
            "s": f"{width}x{height}",
            "pixel_format": self._format.pix_fmt_data,
            "loglevel": self.loglevel,
        }
        f_input = ffmpeg.input("pipe:", **input_args)
        # set output format
        output_args = {
            "vcodec": self._format.codec,
            "pix_fmt": self._format.pix_fmt_file,
            "metadata": "comment=" + shlex.quote(self._get_metadata()),
        }
        if "264" in self._format.codec:
            # set extra options for the H.264 codec
            output_args["crf"] = "0"  # Constant Rate Factor (lower = less compression)
            # make the H.264 codec use the full color range:
            output_args["bsf"] = "h264_metadata=video_full_range_flag=1"
        if self.bitrate > 0:
            # set the specified bitrate
            output_args["video_bitrate"] = str(self.bitrate)
        f_output = f_input.output(filename=self.filename, **output_args)
        self._ffmpeg = f_output.run_async(pipe_stdin=True)  # start process

        if self.write_times:
            self._times_file = self._filename_times.open("w")  # noqa: SIM115

        self.info["num_frames"] = 0
        self._warned_normalization = False
        self._state = "writing"

    def _append_data(self, data: np.ndarray, time: float) -> None:
        """Append a new data set.

        Args:
            data (:class:`~numpy.ndarray`): The actual data
            time (float): The time point associated with the data (currently not used)
        """
        if self._state != "writing" or self._ffmpeg is None:
            RuntimeError(
                "Writing not initialized. Call "
                f"`{self.__class__.__name__}.start_writing`"
            )
        # normalization and video format have been initialized
        assert self._grid is not None
        assert self._norms is not None
        assert self._format is not None

        if self._times_file:
            # write exact time to file
            self._times_file.write(f"{time}\n")
        else:
            # check whether time increases as expected
            t_start = self.info.get("t_start")
            if t_start is None:
                t_start = 0
            dt = self.info.get("dt", 1)
            time_expect = t_start + dt * self.info["num_frames"]
            if not np.isclose(time, time_expect):  # discrepancy in time  # noqa: SIM102
                if not self.info.get("time_mismatch", False):  # not yet warned
                    self._logger.warning("Time mismatch: %g != %g", time, time_expect)
                    self.info["time_mismatch"] = True

        # make sure there are two spatial dimensions
        grid_dim = self._grid.num_axes
        if grid_dim > 2:
            raise NotImplementedError
        if grid_dim == 1:
            data = data.reshape(data.shape + (1,))
        # check spatial dimensions
        assert data.ndim >= 2
        assert data.shape[-2:] == self._frame_shape[:2]

        # ensure the data has the shape extra_dim x width x height
        # where `extra_dim` are separated fields or capture the rank of the field
        if data.ndim == 2:
            data = data.reshape((1,) + data.shape)  # explicitly scalar data
        elif data.ndim > 3:
            # collapse first dimensions, so we have three in total
            data = data.reshape((-1,) + data.shape[-2:])
        assert len(self._frame_shape) == data.ndim == 3
        assert len(self._norms) == data.shape[0]  # same non-spatial dimension

        # map data values to frame values
        frame_data = np.zeros(self._frame_shape, dtype=self._format.dtype)
        for i, norm in enumerate(self._norms):
            if not self._warned_normalization:
                if np.any(data[i, ...] < norm.vmin) or np.any(data[i, ...] > norm.vmax):
                    self._logger.warning(
                        "Data outside range specified by `vmin=%g` and `vmax=%g`",
                        norm.vmin,
                        norm.vmax,
                    )
                self._warned_normalization = True  # only warn once
            data_norm = norm(data[i, ...])
            frame_data[..., i] = self._format.data_to_frame(data_norm)

        # write the new data
        self._ffmpeg.stdin.write(frame_data.tobytes())
        self.info["num_frames"] += 1

    def end_writing(self) -> None:
        """Finalize the storage after writing."""
        if self._state != "writing":
            self._logger.warning("Writing was already terminated")
            return  # writing mode was already ended
        self._logger.debug("End writing")
        self.close()

    def __len__(self):
        """Return the number of stored items, i.e., time steps."""
        if "num_frames" not in self.info:
            self._read_metadata()
        return self.info["num_frames"]

    @cached_property()
    def times(self):
        """:class:`~numpy.ndarray`: The times at which data is available."""
        times = None
        if "video_format" not in self.info:
            self._read_metadata()

        if self.info.get("write_times", False):
            # video announces that timestamps have been written to file
            try:
                times = np.loadtxt(self._filename_times)
            except OSError:
                self._logger.warning(
                    "Could not read time stamps from file `%s`. "
                    "Return equidistant times instead.",
                    self._filename_times,
                )

        if times is None:
            # The times need to be re-created from the start and dt information
            t_start = self.info.get("t_start")
            if t_start is None:
                t_start = 0
            dt = self.info.get("dt", 1)
            times = t_start + dt * np.arange(len(self))

        return times

    def _iter_data(self) -> Iterator[np.ndarray]:
        """Iterate over all stored fields."""
        ffmpeg = _import_ffmpeg()  # lazy loading so it's not a hard dependence

        if "width" not in self.info:
            self._read_metadata()
        if self._field is None:
            self._init_field()
        assert self._field is not None
        self._init_normalization(self._field)
        assert self._norms is not None
        frame_shape = (self.info["width"], self.info["height"], self._format.channels)
        data_shape = (len(self._norms), self.info["width"], self.info["height"])
        data = np.empty(data_shape, dtype=self._dtype)
        frame_bytes = np.prod(frame_shape) * self._format.bytes_per_channel

        # iterate over entire movie
        f_input = ffmpeg.input(self.filename, loglevel=self.loglevel)
        f_output = f_input.output(
            "pipe:", format="rawvideo", pix_fmt=self._format.pix_fmt_data
        )
        proc = f_output.run_async(pipe_stdout=True)
        while True:
            read_bytes = proc.stdout.read(frame_bytes)
            if not read_bytes:
                break
            frame = np.frombuffer(read_bytes, self._format.dtype).reshape(frame_shape)

            for i, norm in enumerate(self._norms):
                frame_data = self._format.data_from_frame(frame[:, :, i])
                data[i, :, :] = norm.inverse(frame_data)

            yield data

    @property
    def data(self):
        """:class:`~numpy.ndarray`: The actual data for all times."""
        it = self._iter_data()  # get the iterator of all data
        first_frame = next(it)  # get the first frame to obtain necessary information
        # allocate memory for all data
        data = np.empty((len(self),) + first_frame.shape, dtype=first_frame.dtype)
        data[0] = first_frame  # set the first frame
        for i, frame_data in enumerate(it, 1):  # set all subsequent frames
            data[i] = frame_data
        return data

    def _get_field(self, t_index: int) -> FieldBase:
        """Return the field corresponding to the given time index.

        Load the data given an index, i.e., the data at time `self.times[t_index]`.

        Args:
            t_index (int):
                The index of the data to load

        Returns:
            :class:`~pde.fields.FieldBase`:
            The field class containing the grid and data
        """
        ffmpeg = _import_ffmpeg()  # lazy loading so it's not a hard dependence

        if t_index < 0:
            t_index += len(self)

        if not 0 <= t_index < len(self):
            raise IndexError("Time index out of range")

        if "width" not in self.info:
            self._read_metadata()
        if self._field is None:
            self._init_field()
        assert self._field is not None
        self._init_normalization(self._field)
        assert self._norms is not None
        frame_shape = (self.info["width"], self.info["height"], self._format.channels)
        data_shape = (len(self._norms), self.info["width"], self.info["height"])
        data = np.empty(data_shape, dtype=self._dtype)

        # iterate over entire movie
        f_input = ffmpeg.input(self.filename, loglevel=self.loglevel)
        f_input = f_input.filter("select", f"gte(n,{t_index})")
        f_output = f_input.output(
            "pipe:", vframes=1, format="rawvideo", pix_fmt=self._format.pix_fmt_data
        )
        read_bytes, _ = f_output.run(capture_stdout=True)
        if not read_bytes:
            raise OSError("Could not read any data")
        frame = np.frombuffer(read_bytes, self._format.dtype).reshape(frame_shape)

        for i, norm in enumerate(self._norms):
            frame_data = self._format.data_from_frame(frame[:, :, i])
            data[i, :, :] = norm.inverse(frame_data)

        # create the field with the data of the given index
        assert self._field is not None
        field = self._field.copy()
        field.data = data.reshape(field.data.shape)
        return field

    def __iter__(self) -> Iterator[FieldBase]:
        """Iterate over all stored fields."""
        for data in self._iter_data():
            # create the field with the data of the given index
            assert self._field is not None
            field = self._field.copy()
            field.data = data.reshape(field.data.shape)
            yield field

    def items(self) -> Iterator[tuple[float, FieldBase]]:
        """Iterate over all times and stored fields, returning pairs."""
        yield from zip(self.times, self)

    @fill_in_docstring
    def tracker(
        self,
        interrupts: InterruptData = 1,
        *,
        transformation: Callable[[FieldBase, float], FieldBase] | None = None,
    ) -> StorageTracker:
        """Create object that can be used as a tracker to fill this storage.

        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
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
        interrupts = parse_interrupt(interrupts)

        if not (self.write_times or isinstance(interrupts, ConstantInterrupts)):
            self._logger.warning(
                "Use `write_times=True` to write times for complex interrupts"
            )
        # store data for common case of constant intervals
        self.info["dt"] = getattr(interrupts, "dt", 1)
        self.info["t_start"] = getattr(interrupts, "t_start", 0)

        return StorageTracker(
            storage=self, interrupts=interrupts, transformation=transformation
        )
