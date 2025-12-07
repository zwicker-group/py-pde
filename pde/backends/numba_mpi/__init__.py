"""Defines the :mod:`numba_mpi` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaMPIBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

try:
    import numba_mpi
except ImportError as err:
    msg = (
        "The `numba_mpi` package is required to use the numba_mpi backend. "
        "Please install it using: pip install numba-mpi"
    )
    raise ImportError(msg) from err
else:
    del numba_mpi  # clean workspace

from .. import backends
from .backend import NumbaMPIBackend

# add the loaded numba backend to the registry
backends.add(NumbaMPIBackend("numba_mpi"))

# register some numba overloads
from . import overloads

del overloads
