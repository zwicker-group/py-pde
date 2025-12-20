"""Defines the :mod:`torch` backend.

.. autosummary::
   :nosignatures:

   ~backend.TorchBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import TorchBackend

# add the loaded numba backend to the registry
torch_backend = TorchBackend("torch")
backends.add(torch_backend)

# register all the standard operators
from . import operators

del operators
