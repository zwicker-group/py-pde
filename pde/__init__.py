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
from .tools.config import Config, Parameter, environment  # noqa: F401

config = Config()  # initialize the default configuration

import contextlib

# import all other modules that should occupy the main name space
from .fields import *  # noqa: F403
from .grids import *  # noqa: F403
from .pdes import *  # noqa: F403
from .solvers import *  # noqa: F403
from .storage import *  # noqa: F403
from .trackers import *  # noqa: F403
from .visualization import *  # noqa: F403

with contextlib.suppress(ImportError):
    from .tools.modelrunner import *  # noqa: F403

del contextlib, Config  # clean name space
