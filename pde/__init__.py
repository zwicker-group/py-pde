"""
The py-pde package provides classes and methods for solving partial differential
equations.
"""

# initialize the configuration
from .tools.config import Config, environment

config = Config()  # initialize the default configuration
del Config  # clean the name space


# import all other modules that should occupy the main name space
from .fields import *  # @UnusedWildImport
from .grids import *  # @UnusedWildImport
from .pdes import *  # @UnusedWildImport
from .solvers import *  # @UnusedWildImport
from .storage import *  # @UnusedWildImport
from .tools.parameters import Parameter
from .trackers import *  # @UnusedWildImport
from .version import __version__
from .visualization import *  # @UnusedWildImport
