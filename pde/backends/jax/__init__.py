"""Defines the :mod:`jax` backend.

.. autosummary::
   :nosignatures:

   ~backend.JaxBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import JaxBackend

# add the jax backend class to the registry
backend_registry.register_class("jax", JaxBackend)

# register all the standard operators
from . import operators

del operators
