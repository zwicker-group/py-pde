"""The py-pde package provides classes and methods for solving partial differential
equations."""

# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__
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

import contextlib

# import all other modules that should occupy the main name space
from .fields import *
from .grids import *
from .pdes import *
from .solvers import *
from .storage import *
from .tools.parameters import Parameter
from .trackers import *
from .visualization import *

with contextlib.suppress(ImportError):
    from .tools.modelrunner import *

del contextlib, Config  # clean name space
