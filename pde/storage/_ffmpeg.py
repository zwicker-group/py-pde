"""
Functions for interacting with FFmpeg

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# import subprocess as sp
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

# def _run_ffmpeg(args: list[str]):
#     return sp.check_output(["ffmpeg"] + args)
#
#
# def codecs() -> list[str]:
#     """list: all supported ffmpeg codecs"""
#     res = _run_ffmpeg(["-codecs"])
#
#
# def get_pixel_formats(encoder=None):
#     if encoder is None:
#         res = _run_ffmpeg(["-pix_fmts"])
#     else:
#         res = _run_ffmpeg(["-h", f"encoder={encoder}"])


@dataclass
class FFmpegFormat:
    """defines a FFmpeg format used for storing field data in a video"""

    pix_fmt_file: str
    codec: str
    channels: int
    value_max: int
    dtype: DTypeLike

    @property
    def pix_fmt_data(self) -> str:
        """return a suitable pixel format for the field data"""
        if self.channels == 1:
            return "gray"
        elif self.channels == 3:
            return "rgb24"
        elif self.channels == 4:
            return "rgba"
        else:
            raise NotImplementedError(f"Cannot deal with {self.channels} channels")

    def data_to_frame(self, normalized_data: np.ndarray) -> np.ndarray:
        return (normalized_data * self.value_max).astype(self.dtype)

    def data_from_frame(self, frame_data: np.ndarray):
        return frame_data.astype(float) / self.value_max


formats = {
    "gray": FFmpegFormat(
        pix_fmt_file="gray",
        codec="libx264",
        channels=1,
        value_max=255,
        dtype=np.uint8,
    ),
    "rgb24": FFmpegFormat(
        pix_fmt_file="rgb24",
        codec="libx264rgb",
        channels=3,
        value_max=255,
        dtype=np.uint8,
    ),
    "bgr24": FFmpegFormat(
        pix_fmt_file="bgr24",
        codec="libx264rgb",
        channels=3,
        value_max=255,
        dtype=np.uint8,
    ),
    "rgb32": FFmpegFormat(
        pix_fmt_file="rgb32",
        codec="libx264rgb",
        channels=4,
        value_max=255,
        dtype=np.uint8,
    ),
}
