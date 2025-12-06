"""Defines the :mod:`numba` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import NumbaBackend

# add the loaded numba backend to the registry
numba_backend = NumbaBackend("numba")
backends.add(numba_backend)

# register all the standard operators
from . import operators, overloads

del operators, overloads
