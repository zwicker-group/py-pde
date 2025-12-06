"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from ..numba.backend import NumbaBackend


class NumbaMPIBackend(NumbaBackend):
    """Defines MPI-compatible numba backend."""
