"""Module defining classes for storing simulation data.

.. autosummary::
   :nosignatures:

   ~memory.get_memory_storage
   ~memory.MemoryStorage
   ~modelrunner.ModelrunnerStorage
   ~file.FileStorage
   ~movie.MovieStorage

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import contextlib

from .file import FileStorage
from .memory import MemoryStorage, get_memory_storage
from .movie import MovieStorage

with contextlib.suppress(ImportError):
    from .modelrunner import ModelrunnerStorage
