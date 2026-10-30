"""Defines a class storing data on the file system as a compressed movie.

This package requires the optional :mod:`av` package to use FFmpeg for reading and
writing movies.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import shlex
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from ..fields import FieldCollection
from ..tools import ffmpeg as FFmpeg
from ..tools.cache import cached_property
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from ..trackers.interrupts import ConstantInterrupts, InterruptData, parse_interrupt
from .base import InfoDict, StorageBase, StorageTracker, WriteModeType

_MOV_PIXEL_FORMATS = {
    "gray": "gray",
    "rgb24": "rgb24",
    "rgb32": "rgba",
    "gray16le": "gray16be",
    "gbrp16le": "rgb48be",
    "gbrap16le": "rgba64be",
}
_FFV1_PIXEL_FORMATS = {"rgb24": "bgr0"}


if TYPE_CHECKING:
    import io
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from matplotlib.colors import Normalize
    from numpy.typing import ArrayLike

    from ..fields.base import FieldBase
    from ..fields.datafield_base import DataFieldBase
    from ..tools.typing import NumericArray


def _get_limits(value: float | ArrayLike, dim: int) -> NumericArray:
    """Helper function creating sequence of length `dim` from input."""
    if np.isscalar(value):
        return np.full(dim, value, dtype=float)
    return np.asarray(value)[:dim].astype(float)


def _import_av() -> ModuleType:
    """Import PyAV lazily."""
    import av

    return av


class MovieStorage(StorageBase):
    """Store discretized fields in a movie file.

    This storage requires the optional :mod:`av` package.
    The default codec is `FFV1 <https://en.m.wikipedia.org/wiki/FFV1>`_, which supports
    lossless compression for various configurations. Not all video players support this
    codec, but `VLC <https://www.videolan.org>`_ usually works quite well.
    The MOV container uses PNG encoding because its muxer does not advertise FFV1
    support through PyAV.

    The file extension determines the container format. The container formats ".avi"
    (default), ".mkv", and ".mov" are tested and supported. Other container formats
    supported by `ffmpeg` can be used, but they are not tested and might not work. For
    instance, ".webm" does not support the FFV1 codec and ".nut" files cannot be read
    back.

    Note that important meta information is stored as a comment in the movie, so this
    data must not be deleted or altered if the video should be read again. This
    information includes the version of the storage format. Movies without version
    information are assumed to have format version 1 and movies with unknown format
    versions cannot be read.

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

    _format_version: int = 1

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
                container format of the movie. FFV1 is used for AVI and Matroska files;
                MOV files use lossless PNG encoding.
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
                Legacy option. PyAV logging is process-wide and should instead be
                configured with :func:`av.logging.set_level`.
        """
        # We need to define some quantities early to ensure a graceful cleanup since
        # the `close` method may be called in case of errors in __init__.
        self._container: Any = None
        self._stream: Any = None
        self._encoder_pixel_format: str | None = None
        self._times_file: io.TextIOBase | None = None
        if not module_available("av"):
            msg = "`MovieStorage` needs the `av` package"
            raise ModuleNotFoundError(msg)

        super().__init__(info=info, write_mode=write_mode)
        self.filename = Path(filename)
        self.vmin = vmin
        self.vmax = vmax
        self.bits_per_channel = bits_per_channel
        self.video_format = video_format
        self.bitrate = bitrate
        self.loglevel = loglevel
        if loglevel != "warning":
            warnings.warn(
                "`MovieStorage` does not apply `loglevel` with PyAV. Configure PyAV "
                "logging with `av.logging.set_level` instead.",
                UserWarning,
                stacklevel=2,
            )
        self.write_times = write_times

        self._state: Literal["closed", "writing"] = "closed"
        self._norms: list[Normalize] | None = None

    def __del__(self):
        self.close()  # ensure open files are closed when the FileStorage is deleted

    def close(self) -> None:
        """Close the currently opened file."""
        if self._container is not None:
            self._logger.info("Close movie file '%s'", self.filename)
            try:
                if self._state == "writing":
                    assert self._stream is not None
                    for packet in self._stream.encode():
                        self._container.mux(packet)
            finally:
                self._container.close()
                self._container = None
                self._stream = None

        if self._times_file is not None:
            self._times_file.close()
            self._times_file = None

        self._state = "closed"

    def __enter__(self) -> Self:
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
        info["version"] = self._format_version
        info["vmin"] = self.vmin
        info["vmax"] = self.vmax
        info["write_times"] = self.write_times
        return json.dumps(info)

    def _read_metadata(self) -> None:
        """Read metadata from video and store it in :attr:`info`"""
        av = _import_av()

        path = Path(self.filename)
        if not path.exists():
            msg = f"File `{path}` does not exist"
            raise OSError(msg)
        with av.open(str(path)) as container:
            # sanity checks on the video
            streams = container.streams.video
            if len(container.streams) != 1:
                self._logger.warning(
                    "Only using first of %d streams", len(container.streams)
                )
            if not streams:
                msg = "Movie does not contain a video stream"
                raise RuntimeError(msg)
            stream = streams[0]

            # Read comment field, which can be either lower case or upper case.
            raw_comment = container.metadata.get(
                "comment", container.metadata.get("COMMENT", "{}")
            )
            if raw_comment == "{}":
                self._logger.warning("Could not find metadata written by `py-pde`")
            try:
                metadata = json.loads(raw_comment)
            except json.JSONDecodeError:
                metadata = json.loads(shlex.split(raw_comment)[0])

            version = metadata.pop("version", 1)
            if version != self._format_version:
                msg = (
                    f"Cannot read `{self.filename}` with unknown format version "
                    f"`{version}` (supported: {self._format_version})"
                )
                raise RuntimeError(msg)
            self.vmin = metadata.pop("vmin", 0)
            self.vmax = metadata.pop("vmax", 1)
            self.info.update(metadata)

            num_frames = stream.frames
            if num_frames <= 0:
                # Some containers do not report the frame count in their header.
                num_frames = sum(1 for _ in container.decode(stream))
            self.info["num_frames"] = num_frames
            self.info["width"] = stream.width
            self.info["height"] = stream.height
            pix_fmt = stream.codec_context.format
            pix_fmt_name = pix_fmt.name if pix_fmt is not None else None

        if self.video_format == "auto":
            video_format = self.info.get("video_format")
            if video_format is None:
                video_format = pix_fmt_name
            if video_format is None:
                msg = "Could not determine video format from file"
                raise RuntimeError(msg)
        else:
            video_format = self.video_format
        try:
            self._format = FFmpeg.formats[video_format]
        except KeyError:
            self._logger.warning("Unknown pixel format `%s`", video_format)
        else:
            if self._format.pix_fmt_file != pix_fmt_name:
                self._logger.info(
                    "Pixel format differs from requested one: %s != %s",
                    self._format.pix_fmt_file,
                    pix_fmt_name,
                )

    def _init_normalization(self, field: FieldBase) -> None:
        """Initialize the normalizations of the color information.

        Args:
            field (:class:`~pde.fields.base.FieldBase):
                Example field to obtain information about grid and data rank

        The resulting normalization functions are stored in `self._norms`
        """
        from matplotlib.colors import Normalize

        self._norms = []
        if isinstance(field, FieldCollection):
            fields: Sequence[DataFieldBase] = field.fields
        else:
            fields = [field]  # type: ignore
        vmin = _get_limits(self.vmin, len(fields))
        vmax = _get_limits(self.vmax, len(fields))
        for f_id, f in enumerate(fields):
            norm = Normalize(vmin[f_id], vmax[f_id], clip=True)
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
        if self._state == "writing":
            msg = f"{self.__class__.__name__} is already in writing mode"
            raise RuntimeError(msg)
        if self._container is not None:
            msg = "Movie container already started"
            raise RuntimeError(msg)

        # delete data if truncation is requested. This is for instance necessary
        # to remove older data with incompatible data_shape
        if self.write_mode == "truncate":
            self.clear()
        elif self.write_mode == "truncate_once":
            self.clear()
            self.write_mode = "append"  # do not truncate in subsequent calls
        elif self.write_mode == "append":
            msg = "Appending to movies is not possible"
            raise NotImplementedError(msg)

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
            msg = "Cannot use grid with more than two axes"
            raise RuntimeError(msg)

        # get color channel information
        if self.video_format == "auto":
            channels = field._data_flat.shape[0]
            video_format = FFmpeg.find_format(channels, self.bits_per_channel)
            if video_format is None:
                msg = (
                    f"Could not find a video format with {channels} channels and "
                    f"{self.bits_per_channel} bits per channel."
                )
                raise RuntimeError(msg)
            self.info["video_format"] = video_format
        else:
            self.info["video_format"] = self.video_format
        self._format = FFmpeg.formats[self.info["video_format"]]
        if field.is_complex:
            msg = "MovieStorage does not support complex values"
            raise NotImplementedError(msg)
        self._frame_shape = (width, height, self._format.channels)

        # set up the normalization
        self._init_normalization(field)

        av = _import_av()
        self._logger.debug("Start PyAV writer for `%s`", self.filename)
        self._container = av.open(str(self.filename), mode="w")
        codec = self._format.codec
        if self.filename.suffix.lower() == ".mov" and codec == "ffv1":
            codec = "png"
            self._encoder_pixel_format = _MOV_PIXEL_FORMATS[self.info["video_format"]]
        else:
            self._encoder_pixel_format = _FFV1_PIXEL_FORMATS.get(
                self._format.pix_fmt_file, self._format.pix_fmt_file
            )
        self._stream = self._container.add_stream(codec, rate=25)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = self._encoder_pixel_format
        if self.bitrate > 0:
            self._stream.bit_rate = self.bitrate
        self._container.metadata["comment"] = self._get_metadata()

        if self.write_times:
            self._times_file = self._filename_times.open("w")

        self.info["num_frames"] = 0
        self._warned_normalization = False
        self._state = "writing"

    def _append_data(self, data: NumericArray, time: float) -> None:
        """Append a new data set.

        Args:
            data (:class:`~numpy.ndarray`): The actual data
            time (float): The time point associated with the data
        """
        if self._state != "writing" or self._container is None or self._stream is None:
            msg = (
                "Writing not initialized. Call "
                f"`{self.__class__.__name__}.start_writing`"
            )
            raise RuntimeError(msg)
        # normalization and video format have been initialized
        assert self._grid is not None
        assert self._norms is not None
        assert self._format is not None
        assert self._encoder_pixel_format is not None

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
            data = data.reshape((*data.shape, 1))
        # check spatial dimensions
        assert data.ndim >= 2
        assert data.shape[-2:] == self._frame_shape[:2]

        # ensure the data has the shape extra_dim x width x height
        # where `extra_dim` are separated fields or capture the rank of the field
        if data.ndim == 2:
            data = data.reshape((1, *data.shape))  # explicitly scalar data
        elif data.ndim > 3:
            # collapse first dimensions, so we have three in total
            data = data.reshape((-1, *data.shape[-2:]))
        assert len(self._frame_shape) == data.ndim == 3
        assert len(self._norms) == data.shape[0]  # same non-spatial dimension

        # map data values to frame values
        frame_data = np.zeros(self._frame_shape, dtype=self._format.dtype)
        for i, norm in enumerate(self._norms):
            if not self._warned_normalization:
                if np.any(data[i, ...] < norm.vmin) or np.any(data[i, ...] > norm.vmax):  # type: ignore
                    self._logger.warning(
                        "Data outside range specified by `vmin=%g` and `vmax=%g`",
                        norm.vmin,
                        norm.vmax,
                    )
                self._warned_normalization = True  # only warn once
            data_norm = norm(data[i, ...])
            frame_data[..., i] = self._format.data_to_frame(data_norm)

        # write the new data
        av = _import_av()
        frame_array = frame_data.reshape(
            self._frame_shape[1], self._frame_shape[0], self._format.channels
        )
        if self._format.channels == 1:
            frame_array = frame_array[:, :, 0]
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame_array), format=self._format.pix_fmt_data
        )
        if frame.format.name != self._encoder_pixel_format:
            frame = frame.reformat(format=self._encoder_pixel_format)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
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

    def _iter_data(self) -> Iterator[NumericArray]:
        """Iterate over all stored fields."""
        av = _import_av()

        if "width" not in self.info:
            self._read_metadata()
        if self._field is None:
            self._init_field()
        assert self._field is not None
        self._init_normalization(self._field)
        assert self._norms is not None
        data_shape = (len(self._norms), self.info["width"], self.info["height"])
        data = np.empty(data_shape, dtype=self._dtype)

        with av.open(str(self.filename)) as container:
            stream = container.streams.video[0]
            for video_frame in container.decode(stream):
                frame = video_frame.to_ndarray(format=self._format.pix_fmt_data)
                if frame.ndim == 2:
                    frame = frame[:, :, np.newaxis]
                frame = frame.reshape(
                    self.info["width"], self.info["height"], self._format.channels
                )

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
        data = np.empty((len(self), *first_frame.shape), dtype=first_frame.dtype)
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
        av = _import_av()

        if t_index < 0:
            t_index += len(self)

        if not 0 <= t_index < len(self):
            msg = "Time index out of range"
            raise IndexError(msg)

        if "width" not in self.info:
            self._read_metadata()
        if self._field is None:
            self._init_field()
        assert self._field is not None
        self._init_normalization(self._field)
        assert self._norms is not None
        data_shape = (len(self._norms), self.info["width"], self.info["height"])
        data = np.empty(data_shape, dtype=self._dtype)

        with av.open(str(self.filename)) as container:
            stream = container.streams.video[0]
            video_frame = next(
                (
                    frame
                    for i, frame in enumerate(container.decode(stream))
                    if i == t_index
                ),
                None,
            )
        if video_frame is None:
            msg = "Could not read any data"
            raise OSError(msg)
        frame = video_frame.to_ndarray(format=self._format.pix_fmt_data)
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        frame = frame.reshape(
            self.info["width"], self.info["height"], self._format.channels
        )

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
        yield from zip(self.times, self, strict=False)

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
