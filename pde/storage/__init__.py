"""
Module defining classes for storing simulation data.

.. autosummary::
   :nosignatures:

   ~memory.get_memory_storage
   ~memory.MemoryStorage
   ~modelrunner.ModelrunnerStorage
   ~file.FileStorage
   ~movie.MovieStorage

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from .file import FileStorage
from .memory import MemoryStorage, get_memory_storage
from .movie import MovieStorage

try:
    from .modelrunner import ModelrunnerStorage
except ImportError:
    ...  # ModelrunnerStorage is only available when py-modelrunner is available
