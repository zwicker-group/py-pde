"""
Functions for interacting with FFmpeg

.. autosummary::
   :nosignatures:

   FFmpegFormat
   formats
   find_format

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike


@dataclass
class FFmpegFormat:
    """defines a FFmpeg format used for storing field data in a video

    All pixel formats supported by FFmpeg can be obtained by running
    :code:`ffmpeg -pix_fmts`. However, not all pixel formats are supported by all
    codecs. Supported pixel formats are listed in the output of
    :code:`ffmpeg -h encoder=<ENCODER>`, where `<ENCODER>` is one of the encoders listed
    in :code:`ffmpeg -codecs`.
    """

    pix_fmt_file: str
    """str: name of the pixel format used in the codec"""
    channels: int
    """int: number of color channels in this pixel format"""
    bits_per_channel: int
    """int: number of bits per color channel in this pixel format"""
    codec: str = "ffv1"
    """str: name of the codec that supports this pixel format"""

    @property
    def pix_fmt_data(self) -> str:
        """return a suitable pixel format for the field data"""
        if self.bits_per_channel == 8:
            if self.channels == 1:
                return "gray"
            elif self.channels == 3:
                return "rgb24"
            elif self.channels == 4:
                return "rgba"
            else:
                raise NotImplementedError(f"Cannot deal with {self.channels} channels")
        elif self.bits_per_channel == 16:
            if self.channels == 1:
                return "gray16le"
            elif self.channels == 3:
                return "gbrp16le"
            elif self.channels == 4:
                return "rgba64le"
            else:
                raise NotImplementedError(f"Cannot deal with {self.channels} channels")
        else:
            raise NotImplementedError(f"Cannot use {self.bits_per_channel} bits")

    @property
    def bytes_per_channel(self) -> int:
        """int:number of bytes per color channel"""
        return self.bits_per_channel // 8

    @property
    def dtype(self) -> DTypeLike:
        """numpy dtype corresponding to the data of a single channel"""
        if self.bits_per_channel == 8:
            return np.uint8
        elif self.bits_per_channel == 16:
            return np.uint16
        else:
            raise NotImplementedError(f"Cannot use {self.bits_per_channel} bits")

    @property
    def value_max(self) -> int:
        """maximal value stored in a color channel"""
        return 2**self.bits_per_channel - 1  # type: ignore

    def data_to_frame(self, normalized_data: np.ndarray) -> np.ndarray:
        """converts normalized data to data being stored in a color channel"""
        return (normalized_data * self.value_max).astype(self.dtype)

    def data_from_frame(self, frame_data: np.ndarray):
        """converts data stored in a color channel to normalized data"""
        return frame_data.astype(float) / self.value_max


formats = {
    "gray": FFmpegFormat(pix_fmt_file="gray", channels=1, bits_per_channel=8),
    "rgb24": FFmpegFormat(pix_fmt_file="rgb24", channels=3, bits_per_channel=8),
    "rgb32": FFmpegFormat(pix_fmt_file="rgb32", channels=4, bits_per_channel=8),
    "gray16le": FFmpegFormat(pix_fmt_file="gray16le", channels=1, bits_per_channel=16),
    "gbrp16le": FFmpegFormat(pix_fmt_file="gbrp16le", channels=3, bits_per_channel=16),
    "rgba64le": FFmpegFormat(pix_fmt_file="rgba64le", channels=4, bits_per_channel=16),
}
"""dict of :class:`FFmpegFormat` formats"""


def find_format(channels: int, bits_per_channel: int = 8) -> Optional[str]:
    """find a defined FFmpegFormat that satisifies the requirements

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
        if f.channels >= channels and f.bits_per_channel >= bits_per_channel:
            # this format satisfies the requirements
            if (
                f_best is None
                or f.bits_per_channel < f_best.bits_per_channel
                or f.channels < f_best.channels
            ):
                n_best, f_best = n, f
    return n_best
