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
from .tools.config import Config, Parameter, environment  # noqa: F401

config = Config()  # initialize the default configuration

import contextlib

# import most common classes into main name space
from .backends import backends  # noqa: F401
from .fields import *  # noqa: F403
from .grids import *  # noqa: F403
from .pdes import *  # noqa: F403
from .solvers import *  # noqa: F403
from .storage import *  # noqa: F403
from .trackers import *  # noqa: F403
from .visualization import *  # noqa: F403

# try registering the hooks that allow py-modelrunner to store fields from py-pde
with contextlib.suppress(ImportError):
    from .tools.modelrunner import *  # noqa: F403

del contextlib, Config  # clean name space
