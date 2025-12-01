"""Defines the :mod:`numba` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import NumbaBackend

# add the loaded numba backend to the registry
backends.add(NumbaBackend("numba"))

# register all the standard operators
from . import operators  # noqa: F401
