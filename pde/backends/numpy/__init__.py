"""Defines the :mod:`numpy` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumpyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backend_registry
from .backend import NumpyBackend

numpy_backend = NumpyBackend(backend_registry.get_config("numpy"), name="numpy")
backend_registry.add(numpy_backend)
