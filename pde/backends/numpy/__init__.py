"""Defines the :mod:`numpy` backend.

.. autosummary::
   :nosignatures:

   ~backend.NumpyBackend

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .backend import NumpyBackend

numpy_backend = NumpyBackend("numpy")
