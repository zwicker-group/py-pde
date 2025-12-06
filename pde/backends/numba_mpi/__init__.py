"""Defines the :mod:`numba` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaMPIBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import NumbaMPIBackend

# add the loaded numba backend to the registry
backends.add(NumbaMPIBackend("numba_mpi"))
