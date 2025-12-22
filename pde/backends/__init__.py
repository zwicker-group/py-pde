"""Defines backends, which implement efficient numerical simulations.

.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~numba.backend.NumbaBackend
   ~numba_mpi.backend.NumbaMPIBackend
   ~numpy.backend.NumpyBackend
   ~scipy.backend.ScipyBackend
   ~torch.backend.TorchBackend

Inheritance structure of the classes:

.. inheritance-diagram::
         numba.backend.NumbaBackend
         numba_mpi.backend.NumbaMPIBackend
         numpy.backend.NumpyBackend
         scipy.backend.ScipyBackend
         torch.backend.TorchBackend
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pathlib import Path

# load and register the numpy backend, which is the default
from .numpy import numpy_backend

# load the registry, which manages all backends
from .registry import backends, load_default_config

# add the default backend, which will be loaded anytime
backends.add(numpy_backend)

# register additional backends without loading them
BACKENDS_FOLDER = Path(__file__).parent
backends.register_package(
    "numba",
    "pde.backends.numba",
    config=load_default_config(BACKENDS_FOLDER / "numba" / "config.py"),
)
backends.register_package("numba_mpi", "pde.backends.numba_mpi")
backends.register_package("scipy", "pde.backends.scipy")
backends.register_package(
    "torch",
    "pde.backends.torch",
    config=load_default_config(BACKENDS_FOLDER / "torch" / "config.py"),
)

# clean namespace
del Path, load_default_config, BACKENDS_FOLDER
