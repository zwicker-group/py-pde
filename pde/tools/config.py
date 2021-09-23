"""
Handles configuration variables of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import collections
import importlib
import sys
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
        256 ** 2,
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


def environment(dict_type=dict) -> Dict[str, Any]:
    """obtain information about the compute environment

    Args:
        dict_type:
            The type to create the returned dictionaries. The default is `dict`, but
            :class:`collections.OrderedDict` is an alternative.

    Returns:
        dict: information about the python installation and packages
    """
    import matplotlib as mpl

    from .. import __version__ as package_version
    from .. import config
    from .numba import numba_environment
    from .plotting import get_plotting_context

    def get_package_versions(packages: List[str]) -> Dict[str, str]:
        """tries to load certain python packages and returns their version"""
        versions: Dict[str, str] = dict_type()
        for name in sorted(packages):
            try:
                module = importlib.import_module(name)
            except ImportError:
                versions[name] = "not available"
            else:
                versions[name] = module.__version__  # type: ignore
        return versions

    result: Dict[str, Any] = dict_type()
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
