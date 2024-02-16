"""
The py-pde package provides classes and methods for solving partial differential
equations.
"""

# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__  # type: ignore
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pde")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine any version
        __version__ = "unknown"
    del PackageNotFoundError, version  # clean name space

# initialize the configuration
from .tools.config import Config, environment

config = Config()  # initialize the default configuration
del Config  # clean name space

# import all other modules that should occupy the main name space
from .fields import *  # @UnusedWildImport
from .grids import *  # @UnusedWildImport
from .pdes import *  # @UnusedWildImport
from .solvers import *  # @UnusedWildImport
from .storage import *  # @UnusedWildImport
from .tools.parameters import Parameter
from .trackers import *  # @UnusedWildImport
from .visualization import *  # @UnusedWildImport
