"""Defines the :mod:`numpy` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumpyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .. import backends
from .backend import NumpyBackend

numpy_backend = NumpyBackend(backends.get_config("numpy"), name="numpy")
backends.add(numpy_backend)
