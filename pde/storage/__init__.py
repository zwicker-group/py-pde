"""Module defining classes for storing simulation data.

.. autosummary::
   :nosignatures:

   ~base.StorageBase
   ~memory.MemoryStorage
   ~memory.get_memory_storage
   ~file.FileStorage
   ~movie.MovieStorage
   ~modelrunner.ModelrunnerStorage

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
