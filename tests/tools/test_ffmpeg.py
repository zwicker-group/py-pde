"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.tools.ffmpeg import find_format


@pytest.mark.parametrize(
    "channels,bits_per_channel,result",
    [(1, 8, "gray"), (2, 7, "rgb24"), (3, 9, "gbrp16le"), (5, 8, None), (1, 17, None)],
)
def test_find_format(channels, bits_per_channel, result):
    """test_find_format function."""
    assert find_format(channels, bits_per_channel) == result
