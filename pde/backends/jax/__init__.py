"""Defines the :mod:`jax` backend.

.. autosummary::
   :nosignatures:

   ~backend.JaxBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import JaxBackend

# add the loaded jax backend to the registry
jax_backend = JaxBackend(backend_registry.get_config("jax"), name="jax")
backend_registry.add(jax_backend)

# register all the standard operators
from . import operators

del operators
