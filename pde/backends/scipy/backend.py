"""Defines the scipy backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from ..numpy.backend import NumpyBackend


class ScipyBackend(NumpyBackend):
    """Defines scipy backend."""
