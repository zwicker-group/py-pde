"""Defines backends, which implement efficient numerical simulations.

.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~numba.backend.NumbaBackend
   ~numba_mpi.backend.NumbaMPIBackend
   ~numpy.backend.NumpyBackend
   ~scipy.backend.ScipyBackend

Inheritance structure of the classes:

.. inheritance-diagram::
         numba.backend.NumbaBackend
         numba_mpi.backend.NumbaMPIBackend
         numpy.backend.NumpyBackend
         scipy.backend.ScipyBackend
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# load the registry, which manages all backends
from .registry import backends  # noqa: I001

# load and register the numpy backend, which is the default
from .numpy import numpy_backend

backends.add(numpy_backend)

# register additional backends without loading them
backends.register_package("pde.backends.numba", "numba")
backends.register_package("pde.backends.numba_mpi", "numba_mpi")
backends.register_package("pde.backends.scipy", "scipy")
