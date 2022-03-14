"""
Handles configuration variables of the package

.. autosummary::
   :nosignatures:

   Config
   get_package_versions
   parse_version_str
   check_package_version
   environment
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import collections
import importlib
import sys
import warnings
from typing import Any, Dict, List

from .misc import module_available
from .parameters import Parameter

# define default parameter values
DEFAULT_CONFIG: List[Parameter] = [
    Parameter(
        "numba.debug",
        False,
        bool,
        "Determines whether numba used the debug mode for compilation. If enabled, "
        "this emits extra information that might be useful for debugging.",
    ),
    Parameter(
        "numba.fastmath",
        True,
        bool,
        "Determines whether the fastmath flag is set during compilation. This affects "
        "the precision of the mathematical calculations.",
    ),
    Parameter(
        "numba.parallel",
        True,
        bool,
        "Determines whether multiple cores are used in numba-compiled code.",
    ),
    Parameter(
        "numba.parallel_threshold",
        256**2,
        int,
        "Minimal number of support points before multithreading or multiprocessing is "
        "enabled in the numba compilations.",
    ),
]


class Config(collections.UserDict):
    """class handling the package configuration"""

    def __init__(self, items: Dict[str, Any] = None, mode: str = "update"):
        """
        Args:
            items (dict, optional):
                Configuration values that should be added or overwritten to initialize
                the configuration.
            mode (str):
                Defines the mode in which the configuration is used. Possible values are

                * `insert`: any new configuration key can be inserted
                * `update`: only the values of pre-existing items can be updated
                * `locked`: no values can be changed

                Note that the items specified by `items` will always be inserted,
                independent of the `mode`.
        """
        self.mode = "insert"  # temporarily allow inserting items
        super().__init__({p.name: p for p in DEFAULT_CONFIG})
        if items:
            self.update(items)
        self.mode = mode

    def __getitem__(self, key: str):
        """retrieve item `key`"""
        parameter = self.data[key]
        if isinstance(parameter, Parameter):
            return parameter.convert()
        else:
            return parameter

    def __setitem__(self, key: str, value):
        """update item `key` with `value`"""
        if self.mode == "insert":
            self.data[key] = value

        elif self.mode == "update":
            try:
                self[key]  # test whether the key already exist (including magic keys)
            except KeyError:
                raise KeyError(
                    f"{key} is not present and config is not in `insert` mode"
                )
            self.data[key] = value

        elif self.mode == "locked":
            raise RuntimeError("Configuration is locked")

        else:
            raise ValueError(f"Unsupported configuration mode `{self.mode}`")

    def __delitem__(self, key: str):
        """removes item `key`"""
        if self.mode == "insert":
            del self.data[key]
        else:
            raise RuntimeError("Configuration is not in `insert` mode")

    def to_dict(self) -> Dict[str, Any]:
        """convert the configuration to a simple dictionary

        Returns:
            dict: A representation of the configuration in a normal :class:`dict`.
        """
        return {k: v for k, v in self.items()}

    def __repr__(self) -> str:
        """represent the configuration as a string"""
        return f"{self.__class__.__name__}({repr(self.to_dict())})"


def get_package_versions(
    packages: List[str], *, na_str="not available"
) -> Dict[str, str]:
    """tries to load certain python packages and returns their version

    Args:
        packages (list): The names of all packages
        na_str (str): Text to return if package is not available

    Returns:
        dict: Dictionary with version for each package name
    """
    versions: Dict[str, str] = {}
    for name in sorted(packages):
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = na_str
        else:
            versions[name] = module.__version__
    return versions


def parse_version_str(ver_str: str) -> List[int]:
    """helper function converting a version string into a list of integers"""
    result = []
    for token in ver_str.split(".")[:3]:
        try:
            result.append(int(token))
        except ValueError:
            pass
    return result


def check_package_version(package_name: str, min_version: str):
    """checks whether a package has a sufficient version"""

    msg = f"`{package_name}` version {min_version} required for py-pde"
    try:
        # obtain version of the package
        version = importlib.import_module(package_name).__version__

    except ImportError:
        warnings.warn(f"{msg} (but none installed)")

    else:
        # check whether it is installed and works
        if parse_version_str(version) < parse_version_str(min_version):
            warnings.warn(f"{msg} (installed: {version})")


def environment() -> Dict[str, Any]:
    """obtain information about the compute environment

    Returns:
        dict: information about the python installation and packages
    """
    import matplotlib as mpl

    from .. import __version__ as package_version
    from .. import config
    from .numba import numba_environment
    from .plotting import get_plotting_context

    result: Dict[str, Any] = {}
    result["package version"] = package_version
    result["python version"] = sys.version
    result["platform"] = sys.platform

    # add the package configuration
    result["config"] = config.to_dict()

    # add details for mandatory packages
    result["mandatory packages"] = get_package_versions(
        ["matplotlib", "numba", "numpy", "scipy", "sympy"]
    )
    result["matplotlib environment"] = {
        "backend": mpl.get_backend(),
        "plotting context": get_plotting_context().__class__.__name__,
    }

    # add details about optional packages
    result["optional packages"] = get_package_versions(
        ["h5py", "napari", "pandas", "pyfftw", "tqdm"]
    )
    if module_available("numba"):
        result["numba environment"] = numba_environment()

    return result
