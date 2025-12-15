"""Defines the :mod:`scipy` backend.

.. autosummary::
   :nosignatures:

   ~backend.ScipyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import ScipyBackend

# add the loaded scipy backend to the registry
scipy_backend = ScipyBackend("scipy")
backends.add(scipy_backend)

# register all the standard operators
from . import operators

del operators
