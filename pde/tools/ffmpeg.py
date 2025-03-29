"""Functions for interacting with FFmpeg.

.. autosummary::
   :nosignatures:

   FFmpegFormat
   formats
   find_format

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import DTypeLike


@dataclass
class FFmpegFormat:
    """Defines a FFmpeg format used for storing field data in a video.

    Note:
        All pixel formats supported by FFmpeg can be obtained by running
        :code:`ffmpeg -pix_fmts`. However, not all pixel formats are supported by all
        codecs. Supported pixel formats are listed in the output of
        :code:`ffmpeg -h encoder=<ENCODER>`, where `<ENCODER>` is one of the encoders
        listed in :code:`ffmpeg -codecs`.
    """

    pix_fmt_file: str
    """str: name of the pixel format used in the codec"""
    pix_fmt_data: str
    """str: name of the pixel format used in the frame data"""
    channels: int
    """int: number of color channels in this pixel format"""
    bits_per_channel: int
    """int: number of bits per color channel in this pixel format"""
    dtype: DTypeLike
    """Numpy dtype corresponding to the data of a single channel."""
    codec: str = "ffv1"
    """str: name of the codec that supports this pixel format"""

    @property
    def bytes_per_channel(self) -> int:
        """Int:number of bytes per color channel."""
        return self.bits_per_channel // 8

    @property
    def max_value(self) -> float | int:
        """Maximal value stored in a color channel."""
        if np.issubdtype(self.dtype, np.integer):
            return 2**self.bits_per_channel - 1  # type: ignore
        else:
            return 1.0

    def data_to_frame(self, normalized_data: np.ndarray) -> np.ndarray:
        """Converts normalized data to data being stored in a color channel."""
        return np.ascontiguousarray(normalized_data * self.max_value, dtype=self.dtype)  # type:ignore

    def data_from_frame(self, frame_data: np.ndarray):
        """Converts data stored in a color channel to normalized data."""
        return frame_data.astype(float) / self.max_value


formats = {
    # 8 bit formats
    "gray": FFmpegFormat(
        pix_fmt_file="gray",
        pix_fmt_data="gray",
        channels=1,
        bits_per_channel=8,
        dtype=np.uint8,
    ),
    "rgb24": FFmpegFormat(
        pix_fmt_file="rgb24",
        pix_fmt_data="rgb24",
        channels=3,
        bits_per_channel=8,
        dtype=np.uint8,
    ),
    "rgb32": FFmpegFormat(
        pix_fmt_file="rgb32",
        pix_fmt_data="rgb32",
        channels=4,
        bits_per_channel=8,
        dtype=np.uint8,
    ),
    # 16 bit formats
    "gray16le": FFmpegFormat(
        pix_fmt_file="gray16le",
        pix_fmt_data="gray16le",
        channels=1,
        bits_per_channel=16,
        dtype=np.dtype("<u2"),
    ),
    "gbrp16le": FFmpegFormat(
        pix_fmt_file="gbrp16le",
        pix_fmt_data="gbrp16le",
        channels=3,
        bits_per_channel=16,
        dtype=np.dtype("<u2"),
    ),
    "gbrap16le": FFmpegFormat(
        pix_fmt_file="gbrap16le",
        pix_fmt_data="gbrap16le",
        channels=4,
        bits_per_channel=16,
        dtype=np.dtype("<u2"),
    ),
}
"""Dict of pre-defined :class:`FFmpegFormat` formats."""


def find_format(channels: int, bits_per_channel: int = 8) -> str | None:
    """Find a defined FFmpegFormat that satisifies the requirements.

    Args:
        channels (int):
            Minimal number of color channels
        bits_per_channel (int):
            Minimal number of bits per channel

    Returns:
        str: Identifier for a format that satisifies the requirements (but might have
        more channels or more bits per channel then requested. `None` is returned if no
        format can be identified.
    """
    n_best, f_best = None, None
    for n, f in formats.items():  # iterate through all defined formats
        if (
            f.channels >= channels
            and f.bits_per_channel >= bits_per_channel  # satisfies the requirements
            and (
                f_best is None
                or f.bits_per_channel < f_best.bits_per_channel
                or f.channels < f_best.channels
            )  # the current format is better than the previous one
        ):
            n_best, f_best = n, f
    return n_best
