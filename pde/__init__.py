"""The py-pde package provides tools for solving partial differential equations."""

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
from .tools.config import Config, Parameter, environment

config = Config()  # initialize the default configuration

import contextlib

# import most common classes into main name space
from .fields import *
from .grids import *
from .pdes import *
from .solvers import *
from .storage import *
from .trackers import *
from .visualization import *

# try registering the hooks that allow py-modelrunner to store fields from py-pde
with contextlib.suppress(ImportError):
    from .tools.modelrunner import *

del contextlib, Config  # clean name space
