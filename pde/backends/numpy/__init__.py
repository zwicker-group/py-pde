"""Defines the :mod:`numpy` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumpyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import NumpyBackend

backend_registry.register_class("numpy", NumpyBackend)
