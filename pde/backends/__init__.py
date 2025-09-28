"""Defines backends, which implement efficient numerical simulations.

.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~numba.backend.NumbaBackend
   ~scipy.backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .registry import backends  # noqa

# initialize the standard backends
from . import numba, scipy
