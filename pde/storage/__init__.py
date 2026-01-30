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

from ..tools.misc import module_available
from .base import StorageBase  # noqa: F401
from .file import FileStorage
from .memory import MemoryStorage, get_memory_storage
from .movie import MovieStorage

__all__ = ["FileStorage", "MemoryStorage", "MovieStorage", "get_memory_storage"]

# try importing modelrunner, which is optional
if module_available("modelrunner"):
    from .modelrunner import ModelrunnerStorage

    __all__ += ["ModelrunnerStorage"]
