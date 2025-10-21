"""Defines backends, which implement efficient numerical simulations.

.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~numba.backend.NumbaBackend
   ~numpy.backend.NumpyBackend
   ~scipy.backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# load the registry, which manages all backends
from .registry import backends  # noqa: I001

# load and register the default backend
from .numpy.backend import NumpyBackend

backends.add(NumpyBackend("numpy"))

# register the standard backends without loading them
backends.register_package("pde.backends.numba", "numba")
backends.register_package("pde.backends.scipy", "scipy")
