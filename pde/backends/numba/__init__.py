"""Defines the :mod:`numba` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import NumbaBackend

# register the numba backend
numba_backend = NumbaBackend("numba")
backends.add(numba_backend)

# register all the standard operators
from . import operators
