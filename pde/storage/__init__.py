"""Module defining classes for storing simulation data.

.. autosummary::
   :nosignatures:

   ~base.StorageBase
   ~file.FileStorage
   ~memory.get_memory_storage
   ~memory.MemoryStorage
   ~modelrunner.ModelrunnerStorage
   ~movie.MovieStorage

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import StorageBase  # noqa: F401
from .file import FileStorage
from .memory import MemoryStorage, get_memory_storage
from .movie import MovieStorage

__all__ = ["FileStorage", "MemoryStorage", "MovieStorage", "get_memory_storage"]

# try importing modelrunner, which is optional
try:
    from .modelrunner import ModelrunnerStorage
except ImportError:
    pass  # modelrunner does not seem to be available
else:
    __all__ += ["ModelrunnerStorage"]
