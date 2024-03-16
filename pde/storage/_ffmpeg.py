"""
Functions for interacting with ffmpeg

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import subprocess as sp
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike


@dataclass
class FFmpegPixelFormat:
    pix_fmt: str
    channels: int
    value_max: int
    dtype: DTypeLike

    def data_to_frame(self, normalized_data: np.ndarray) -> np.ndarray:
        return (normalized_data * self.value_max).astype(self.dtype)

    def data_from_frame(self, frame_data: np.ndarray):
        return frame_data.astype(float) / self.value_max


pixel_formats = {
    "gray": FFmpegPixelFormat(
        pix_fmt="gray", channels=1, value_max=255, dtype=np.uint8
    ),
    "rgb24": FFmpegPixelFormat(
        pix_fmt="rgb24", channels=3, value_max=255, dtype=np.uint8
    ),
    "bgr24": FFmpegPixelFormat(
        pix_fmt="bgr24", channels=3, value_max=255, dtype=np.uint8
    ),
    "rgb32": FFmpegPixelFormat(
        pix_fmt="rgb32", channels=4, value_max=255, dtype=np.uint8
    ),
    "gbrp": FFmpegPixelFormat(
        pix_fmt="gbrp", channels=4, value_max=255, dtype=np.uint8
    ),
}


def _run_ffmpeg(args: list[str]):
    return sp.check_output(["ffmpeg"] + args)


def codecs() -> list[str]:
    """list: all supported ffmpeg codecs"""
    res = _run_ffmpeg(["-codecs"])


def get_pixel_formats(encoder=None):
    if encoder is None:
        res = _run_ffmpeg(["-pix_fmts"])
    else:
        res = _run_ffmpeg(["-h", f"encoder={encoder}"])
