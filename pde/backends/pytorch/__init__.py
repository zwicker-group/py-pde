"""Defines the :mod:`pytorch` backend.

.. autosummary::
   :nosignatures:

   ~backend.PytorchBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import PytorchBackend

# add the loaded numba backend to the registry
pytorch_backend = PytorchBackend("pytorch")
backends.add(pytorch_backend)

# register all the standard operators
from . import operators

del operators
