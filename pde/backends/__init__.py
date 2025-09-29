"""Defines backends, which implement efficient numerical simulations.

.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~numba.backend.NumbaBackend
   ~scipy.backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .registry import backends  # noqa

# register the standard backends without loading them
backends.register_package("pde.backends.numba", "numba")
backends.register_package("pde.backends.scipy", "scipy")
