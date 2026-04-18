"""Defines the :mod:`scipy` backend.

.. autosummary::
   :nosignatures:

   ~backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import ScipyBackend

# add the loaded scipy backend to the registry
backend_registry.register_class("scipy", ScipyBackend)

# register all the standard operators
from . import operators

del operators
