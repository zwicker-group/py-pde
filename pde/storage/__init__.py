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

from .base import StorageBase  # noqa: F401
from .file import FileStorage
from .memory import MemoryStorage, get_memory_storage  # noqa: F401
from .movie import MovieStorage

__all__ = ["FileStorage", "MemoryStorage", "MovieStorage"]

# try importing modelrunner, which is optional
try:
    from .modelrunner import ModelrunnerStorage
except ImportError:
    pass
else:
    __all__ += ["ModelrunnerStorage"]
