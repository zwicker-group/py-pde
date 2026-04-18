"""Defines the :mod:`numba` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumbaBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import NumbaBackend

# add the loaded numba backend to the registry
backend_registry.register_class("numba", NumbaBackend)
numba_backend = NumbaBackend(backend_registry.get_config("numba"), name="numba")
backend_registry.register_backend(numba_backend, link_config=True)

# register all the standard operators
from . import operators, overloads

del operators, overloads
