"""
The py-pde package provides classes and methods for solving partial differential
equations.
"""

__version__ = "0.8.0"

from .fields import *  # @UnusedWildImport
from .grids import *  # @UnusedWildImport
from .pdes import *  # @UnusedWildImport
from .solvers import *  # @UnusedWildImport
from .storage import *  # @UnusedWildImport
from .trackers import *  # @UnusedWildImport
from .visualization import *  # @UnusedWildImport

from .tools.misc import environment
from .tools.parameters import Parameter
