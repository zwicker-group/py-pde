"""
The py-pde package provides classes and methods for solving partial differential
equations.
"""

from .version import __version__

from .fields import *  # @UnusedWildImport
from .grids import *  # @UnusedWildImport
from .pdes import *  # @UnusedWildImport
from .solvers import *  # @UnusedWildImport
from .storage import *  # @UnusedWildImport
from .trackers import *  # @UnusedWildImport
from .visualization import *  # @UnusedWildImport

from .tools.misc import environment
from .tools.parameters import Parameter
