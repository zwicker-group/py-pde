"""Handles configuration variables of the package.

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

from __future__ import annotations

import collections
import contextlib
import importlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from .misc import module_available
from .parameters import Parameter

# define default parameter values
DEFAULT_CONFIG: list[Parameter] = [
    Parameter(
        "operators.conservative_stencil",
        True,
        bool,
        "Indicates whether conservative stencils should be used for differential "
        "operators on curvilinear grids. Conservative operators ensure mass "
        "conservation at slightly slower computation speed.",
    ),
    Parameter(
        "operators.tensor_symmetry_check",
        True,
        bool,
        "Indicates whether tensor fields are checked for having a suitable form for "
        "evaluating differential operators in curvilinear coordinates where some axes "
        "are assumed to be symmetric. In such cases, some tensor components might need "
        "to vanish, so the result of the operator can be expressed.",
    ),
    Parameter(
        "operators.cartesian.laplacian_2d_corner_weight",
        0.0,
        float,
        "Weighting factor for the corner points of the 2d cartesian Laplacian. The "
        "standard value is zero, which corresponds to the traditional 5-point stencil. "
        "Alternative choices are 1/2 (Oono-Puri stencil) and 1/3 (Patra-Karttunen or "
        "Mehrstellen stencil); see https://en.wikipedia.org/wiki/Nine-point_stencil.",
    ),
    Parameter(
        "boundaries.accept_lists",
        True,
        bool,
        "Indicate whether boundary conditions can be set using the deprecated legacy "
        "format, where conditions for individual axes and sides where set using lists. "
        "If disabled, only the new format using dicts is supported.",
    ),
    Parameter(
        "numba.debug",
        False,
        bool,
        "Determines whether numba uses the debug mode for compilation. If enabled, "
        "this emits extra information that might be useful for debugging.",
    ),
    Parameter(
        "numba.fastmath",
        True,
        bool,
        "Determines whether the fastmath flag is set during compilation. If enabled, "
        "some mathematical operations might be faster, but less precise. This flag "
        "does not affect infinity detection and NaN handling.",
    ),
    Parameter(
        "numba.multithreading",
        True,
        bool,
        "Determines whether multiple threads are used in numba-compiled code. Enabling "
        "this option accelerates a small subset of operators applied to fields defined "
        "on large grids.",
    ),
    Parameter(
        "numba.multithreading_threshold",
        256**2,
        int,
        "Minimal number of support points of grids before multithreading is enabled in "
        "numba compilations. Has no effect when `numba.multithreading` is `False`.",
    ),
]


class Config(collections.UserDict):
    """Class handling the package configuration."""

    def __init__(self, items: dict[str, Any] | None = None, mode: str = "update"):
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
        """Retrieve item `key`"""
        parameter = self.data[key]
        if isinstance(parameter, Parameter):
            return parameter.convert()
        else:
            return parameter

    def __setitem__(self, key: str, value):
        """Update item `key` with `value`"""
        if self.mode == "insert":
            self.data[key] = value

        elif self.mode == "update":
            try:
                self[key]  # test whether the key already exist (including magic keys)
            except KeyError as err:
                raise KeyError(
                    f"{key} is not present and config is not in `insert` mode"
                ) from err
            self.data[key] = value

        elif self.mode == "locked":
            raise RuntimeError("Configuration is locked")

        else:
            raise ValueError(f"Unsupported configuration mode `{self.mode}`")

    def __delitem__(self, key: str):
        """Removes item `key`"""
        if self.mode == "insert":
            del self.data[key]
        else:
            raise RuntimeError("Configuration is not in `insert` mode")

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a simple dictionary.

        Returns:
            dict: A representation of the configuration in a normal :class:`dict`.
        """
        return dict(self.items())

    def __repr__(self) -> str:
        """Represent the configuration as a string."""
        return f"{self.__class__.__name__}({repr(self.to_dict())})"

    @contextlib.contextmanager
    def __call__(self, values: dict[str, Any] | None = None, **kwargs):
        """Context manager temporarily changing the configuration.

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
    packages: list[str], *, na_str="not available"
) -> dict[str, str]:
    """Tries to load certain python packages and returns their version.

    Args:
        packages (list): The names of all packages
        na_str (str): Text to return if package is not available

    Returns:
        dict: Dictionary with version for each package name
    """
    versions: dict[str, str] = {}
    for name in sorted(packages):
        try:
            module = importlib.import_module(name.replace("-", "_"))
        except ImportError:
            versions[name] = na_str
        else:
            versions[name] = module.__version__
    return versions


def parse_version_str(ver_str: str) -> list[int]:
    """Helper function converting a version string into a list of integers."""
    result = []
    for token in ver_str.split(".")[:3]:
        with contextlib.suppress(ValueError):
            result.append(int(token))
    return result


def check_package_version(package_name: str, min_version: str):
    """Checks whether a package has a sufficient version."""

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


def packages_from_requirements(requirements_file: Path | str) -> list[str]:
    """Read package names from a requirements file.

    Args:
        requirements_file (str or :class:`~pathlib.Path`):
            The file from which everything is read

    Returns:
        list of package names
    """
    result = []
    try:
        with Path(requirements_file).open() as fp:
            for line in fp:
                line_s = line.strip()
                if line_s.startswith("#"):
                    continue
                res = re.search(r"[a-zA-Z0-9_\-]+", line_s)
                if res:
                    result.append(res.group(0))
    except FileNotFoundError:
        result.append(f"Could not open {requirements_file:s}")

    return result


def environment() -> dict[str, Any]:
    """Obtain information about the compute environment.

    Returns:
        dict: information about the python installation and packages
    """
    import matplotlib as mpl

    from .. import __version__ as package_version
    from .. import config
    from . import mpi
    from .numba import numba_environment
    from .plotting import get_plotting_context

    RESOURCE_PATH = Path(__file__).resolve().parents[1] / "tools" / "resources"

    result: dict[str, Any] = {}
    result["package version"] = package_version
    result["python version"] = sys.version
    result["platform"] = sys.platform

    # add the package configuration
    result["config"] = config.to_dict()

    # add details for mandatory packages
    packages_min = packages_from_requirements(RESOURCE_PATH / "requirements_basic.txt")
    result["mandatory packages"] = get_package_versions(packages_min)
    result["matplotlib environment"] = {
        "backend": mpl.get_backend(),
        "plotting context": get_plotting_context().__class__.__name__,
    }

    # add information about jupyter environment
    result["jupyter environment"] = get_package_versions(
        [
            "ipykernel",
            "ipywidgets",
            "jupyter_client",
            "jupyter_core",
            "jupyter_server",
            "notebook",
        ]
    )

    # add details about optional packages
    packages = set(packages_from_requirements(RESOURCE_PATH / "requirements_full.txt"))
    packages |= set(packages_from_requirements(RESOURCE_PATH / "requirements_mpi.txt"))
    packages -= set(packages_min)
    result["optional packages"] = get_package_versions(sorted(packages))
    if module_available("numba"):
        result["numba environment"] = numba_environment()

    # add information about MPI environment
    if mpi.initialized:
        result["multiprocessing"] = {"initialized": True, "size": mpi.size}
    else:
        result["multiprocessing"] = {"initialized": False}

    return result
