"""The py-pde package provides tools for solving partial differential equations.

This package provides classes and methods for solving partial differential equations
(PDEs) on various grids using different numerical methods. Key components include:

- **Fields**: Data structures representing scalar, vector, and tensor fields on grids
- **Grids**: Spatial discretizations including Cartesian and curvilinear coordinates
- **PDEs**: Pre-defined PDEs and a framework for defining custom PDEs
- **Solvers**: Time-stepping algorithms for evolving PDEs
- **Trackers**: Tools for monitoring and analyzing simulations
- **Storage**: Methods for storing simulation data
- **Visualization**: Functions for visualizing fields and creating movies

See the submodules for detailed documentation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

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

# initialize the configuration before doing anything else
from .tools.config import Parameter, config, environment  # noqa

# import most common classes into main name space
from .backends import *  # noqa: F403
from .fields import *  # noqa: F403
from .grids import *  # noqa: F403
from .pdes import *  # noqa: F403
from .solvers import *  # noqa: F403
from .storage import *  # noqa: F403
from .trackers import *  # noqa: F403
from .visualization import *  # noqa: F403

# try registering the hooks that allow py-modelrunner to store fields from py-pde
from .tools.modelrunner import register_modelrunner_hooks

register_modelrunner_hooks()
del register_modelrunner_hooks  # clean name space
