"""Defines the :mod:`torch` backend.

.. autosummary::
   :nosignatures:

   ~backend.TorchBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import TorchBackend

# add the loaded torch backend to the registry
backend_registry.register_class("torch", TorchBackend)

# register all the standard operators
from . import operators

del operators
