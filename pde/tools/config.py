"""
Handles configuration variables of the package

.. autosummary::
   :nosignatures:

   Config
   get_package_versions
   parse_version_str
   check_package_version
   packages_from_requirements
   environment
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import collections
import contextlib
import importlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

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
        "numba.multithreading",
        True,
        bool,
        "Determines whether multiple threads are used in numba-compiled code.",
    ),
    Parameter(
        "numba.multithreading_threshold",
        256**2,
        int,
        "Minimal number of support points before multithreading is enabled in numba "
        "compilations.",
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

    def _translate_deprecated_key(self, key: str) -> str:
        """helper function that allows using deprecated config items"""
        # the depreciations have been introduced on 2022-09-04 and are scheduled to be
        # removed after 2023-03-04
        if key == "numba.parallel":
            warnings.warn(
                "Option `numba.parallel` has been renamed to `numba.multithreading`",
                DeprecationWarning,
            )
            return "numba.multithreading"

        elif key == "numba.parallel_threshold":
            warnings.warn(
                "Option `numba.parallel_threshold` has been renamed to "
                "`numba.multithreading_threshold`",
                DeprecationWarning,
            )
            return "numba.multithreading_threshold"

        return key

    def __getitem__(self, key: str):
        """retrieve item `key`"""
        key = self._translate_deprecated_key(key)
        parameter = self.data[key]
        if isinstance(parameter, Parameter):
            return parameter.convert()
        else:
            return parameter

    def __setitem__(self, key: str, value):
        """update item `key` with `value`"""
        key = self._translate_deprecated_key(key)
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
        key = self._translate_deprecated_key(key)
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

    @contextlib.contextmanager
    def __call__(self, values: Dict[str, Any] = None, **kwargs):
        """context manager temporarily changing the configuration

        Args:
            values (dict): New configuration parameters
            **kwargs: New configuration parameters
        """
        data_initial = self.data.copy()  # save old configuration
        # set new configuration
        if values is not None:
            self.data.update(values)
        self.data.update(kwargs)
        yield  # return to caller
        # restore old configuration
        self.data = data_initial


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
            module = importlib.import_module(name.replace("-", "_"))
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


def packages_from_requirements(requirements_file: Union[Path, str]) -> List[str]:
    """read package names from a requirements file

    Args:
        requirements_file (str or :class:`~pathlib.Path`):
            The file from which everything is read

    Returns:
        list of package names
    """
    result = []
    with open(requirements_file) as fp:
        for line in fp:
            line_s = line.strip()
            if line_s.startswith("#"):
                continue
            res = re.search(r"[a-zA-Z0-9_\-]+", line_s)
            if res:
                result.append(res.group(0))
    return result


def environment() -> Dict[str, Any]:
    """obtain information about the compute environment

    Returns:
        dict: information about the python installation and packages
    """
    import matplotlib as mpl

    from .. import __version__ as package_version
    from .. import config
    from . import mpi
    from .numba import numba_environment
    from .plotting import get_plotting_context

    PACKAGE_PATH = Path(__file__).resolve().parents[2]

    result: Dict[str, Any] = {}
    result["package version"] = package_version
    result["python version"] = sys.version
    result["platform"] = sys.platform

    # add the package configuration
    result["config"] = config.to_dict()

    # add details for mandatory packages
    packages_min = packages_from_requirements(PACKAGE_PATH / "requirements.txt")
    result["mandatory packages"] = get_package_versions(packages_min)
    result["matplotlib environment"] = {
        "backend": mpl.get_backend(),
        "plotting context": get_plotting_context().__class__.__name__,
    }

    # add details about optional packages
    tests_folder = PACKAGE_PATH / "tests"
    packages = set(packages_from_requirements(tests_folder / "requirements_full.txt"))
    packages |= set(packages_from_requirements(tests_folder / "requirements_mpi.txt"))
    packages -= set(packages_min)
    result["optional packages"] = get_package_versions(sorted(packages))
    if module_available("numba"):
        result["numba environment"] = numba_environment()

    # add information about MPI environment
    if mpi.size > 1:
        result["multiprocessing"] = {"initialized": True, "size": mpi.size}
    else:
        result["multiprocessing"] = {"initialized": False}

    return result
