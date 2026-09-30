"""Defines backends, which implement efficient numerical simulations.

Backends are classes that provide logic to carry out numerical calculations. In
particular, each of these classes implements various operators. In principle, backend
classes can be defined independently, but we use some inheritance to share logic.

Moreover, we provide a :class:`~pde.backends.registry.BackendRegistry`, which allows
selecting backends by their identifier, so users do not usually need to construct
backend classes. Users should access the registry via the function :func:`get_backend`
to load a backend in their code.


.. autosummary::
   :nosignatures:

   ~registry.BackendRegistry
   ~jax.backend.JaxBackend
   ~numba.backend.NumbaBackend
   ~numba_mpi.backend.NumbaMPIBackend
   ~numpy.backend.NumpyBackend
   ~scipy.backend.ScipyBackend
   ~torch.backend.TorchBackend

Inheritance structure of the classes:

.. inheritance-diagram::
         jax.backend.JaxBackend
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
from .base import BackendBase
from .registry import (
    available_backends,
    backend_registry,
    get_backend,
    load_default_config,
    registered_backends,
)

# register all backends without loading them
BACKENDS_FOLDER = Path(__file__).parent
backend_registry.register_package(
    name="jax",
    package_path="pde.backends.jax",
    config=load_default_config(BACKENDS_FOLDER / "jax" / "config.py"),
    requires=["jax"],
)
backend_registry.register_package(
    name="numba",
    package_path="pde.backends.numba",
    config=load_default_config(BACKENDS_FOLDER / "numba" / "config.py"),
    requires=["numba"],
)
backend_registry.register_package(
    name="numba_mpi",
    package_path="pde.backends.numba_mpi",
    requires=["numba", "numba_mpi"],
)
backend_registry.register_package(
    name="numpy", package_path="pde.backends.numpy", config=None, requires=["numpy"]
)
backend_registry.register_package(
    name="scipy", package_path="pde.backends.scipy", config=None, requires=["scipy"]
)
backend_registry.register_package(
    name="torch",
    package_path="pde.backends.torch",
    config=load_default_config(BACKENDS_FOLDER / "torch" / "config.py"),
    requires=["torch"],
)

__all__ = [
    "BackendBase",
    "available_backends",
    "get_backend",
    "registered_backends",
]
