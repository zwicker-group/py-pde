"""Defines the :mod:`scipy` backend.

.. autosummary::
   :nosignatures:

   ~backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import ScipyBackend

# add the loaded scipy backend to the registry
scipy_backend = ScipyBackend(backend_registry.get_config("scipy"), name="scipy")
backend_registry.add(scipy_backend)

# register all the standard operators
from . import operators

del operators
