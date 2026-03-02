"""Defines backends, which implement efficient numerical simulations.

Backends are classes that provide logic to carry out numerical calculations. In
particular, each of these classes implements various operators. In principle, backend
classes can be defined independently, but we use some inheritance to share logic.

Moreover, we provide :obj:`~pde.backends.backends`, which is an object of type
:class:`~pde.backends.registry.BackendRegistry`. This registry allows selecting backends
by their identifier, so users do not usually need to construct backend classes.


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

# load the registry, which manages all backends
from .registry import backends, load_default_config, registered_backends

# register backends without loading them
BACKENDS_FOLDER = Path(__file__).parent
backends.register_package("numpy", "pde.backends.numpy")
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

__all__ = ["backends", "registered_backends"]
