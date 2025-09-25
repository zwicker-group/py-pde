"""Handles configuration variables of the package.

.. autosummary::
   :nosignatures:

   Parameter
   Config
   get_package_versions
   parse_version_str
   check_package_version
   packages_from_requirements
   get_ffmpeg_version
   is_hpc_environment
   environment

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import logging
import os
import re
import subprocess as sp
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .misc import module_available


class Parameter:
    """Class representing a single parameter."""

    def __init__(
        self,
        name: str,
        default_value=None,
        cls=object,
        description: str = "",
        hidden: bool = False,
        extra: dict[str, Any] | None = None,
    ):
        """Initialize a parameter.

        Args:
            name (str):
                The name of the parameter
            default_value:
                The default value
            cls:
                The type of the parameter, which is used for conversion
            description (str):
                A string describing the impact of this parameter. This
                description appears in the parameter help
            hidden (bool):
                Whether the parameter is hidden in the description summary
            extra (dict):
                Extra arguments that are stored with the parameter
        """
        self.name = name
        self.default_value = default_value
        self.cls = cls
        self.description = description
        self.hidden = hidden
        self.extra = {} if extra is None else extra

        if cls is not object:
            # check whether the default value is of the correct type
            converted_value = cls(default_value)
            if isinstance(converted_value, np.ndarray):
                # numpy arrays are checked for each individual value
                valid_default = np.allclose(
                    converted_value, default_value, equal_nan=True
                )

            else:
                # other values are compared directly. Note that we also check identity
                # to capture the case where the value is `math.nan`, where the direct
                # comparison (nan == nan) would evaluate to False
                valid_default = (
                    converted_value is default_value or converted_value == default_value
                )

            if not valid_default:
                if hasattr(self, "_logger"):
                    logger: logging.Logger = self._logger
                else:
                    logger = logging.getLogger(self.__class__.__module__)
                logger.warning(
                    "Default value `%s` does not seem to be of type `%s`",
                    name,
                    cls.__name__,
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(name="{self.name}", default_value='
            f'"{self.default_value}", cls="{self.cls.__name__}", '
            f'description="{self.description}", hidden={self.hidden})'
        )

    __str__ = __repr__

    def convert(self, value=None):
        """Converts a `value` into the correct type for this parameter. If `value` is
        not given, the default value is converted.

        Note that this does not make a copy of the values, which could lead to
        unexpected effects where the default value is changed by an instance.

        Args:
            value: The value to convert

        Returns:
            The converted value, which is of type `self.cls`
        """
        if value is None:
            value = self.default_value

        if self.cls is object:
            return value
        else:
            try:
                return self.cls(value)
            except ValueError as err:
                raise ValueError(
                    f"Could not convert {value!r} to {self.cls.__name__} for parameter "
                    f"'{self.name}'"
                ) from err


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
        "operators.cartesian.default_backend",
        "auto",
        str,
        "Sets default backend with which Cartesian operators are created. Typical "
        "options include `scipy` and `numba`. The default option `auto` typically "
        "provides the most flexible result and should rarely be changed.",
    ),
    Parameter(
        "operators.cartesian.laplacian_2d_corner_weight",
        0.0,
        float,
        "Weighting factor for the corner points of the 2d cartesian Laplacian stencil. "
        "The standard value is zero, corresponding to the traditional 5-point stencil. "
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
        "only_local",
        str,
        "Determines whether multiple threads are used in numba-compiled code. Enabling "
        "this option accelerates a small subset of operators applied to fields defined "
        "on large grids. Possible options are 'never' (disable multithreading), "
        "'only_local' (disable on HPC hardware), and 'always' (enable if number of "
        "grid points exceeds `numba.multithreading_threshold`)",
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

    def _convert_value(self, key: str, value):
        """Helper function converting certain values."""
        if key == "numba.multithreading" and isinstance(value, bool):
            value = "always" if value else "never"
            # Deprecated on 2025-02-12
            warnings.warn(
                "Boolean options are deprecated for `numba.multithreading`. Use "
                f"config['numba.multithreading'] = '{value}' instead.",
                DeprecationWarning,
            )
        return value

    def __setitem__(self, key: str, value):
        """Update item `key` with `value`"""
        if self.mode == "insert":
            self.data[key] = self._convert_value(key, value)

        elif self.mode == "update":
            try:
                self[key]  # test whether the key already exist (including magic keys)
            except KeyError as err:
                raise KeyError(
                    f"{key} is not present and config is not in `insert` mode"
                ) from err
            self.data[key] = self._convert_value(key, value)

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

    def use_multithreading(self) -> bool:
        """Determine whether multithreading should be used in numba-compiled code.

        This method checks the configuration setting for `numba.multithreading` and
        determines whether multithreading should be enabled based on the value of this
        setting. The possible values for `numba.multithreading` are:
        - 'always': Multithreading is always enabled.
        - 'never': Multithreading is never enabled.
        - 'only_local': Multithreading is enabled only if the code is not running in a
        high-performance computing (HPC) environment.

        Returns:
            bool: True if multithreading should be enabled, False otherwise.

        Raises:
            ValueError: If the `numba.multithreading` setting is not one of the expected
            values ('always', 'never', 'only_local').
        """
        setting = self["numba.multithreading"]
        if setting == "always":
            return True
        elif setting == "never":
            return False
        elif setting == "only_local":
            return not is_hpc_environment()
        else:
            raise ValueError(
                "Parameter `numba.multithreading` must be in {'always', 'never', "
                f"'only_local'}}, not `{setting}`"
            )


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
            version = importlib.metadata.version(name)
        except ImportError:
            versions[name] = na_str
        else:
            versions[name] = version
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


def get_ffmpeg_version() -> str | None:
    """Read version number of ffmpeg program."""
    # run ffmpeg to get its version
    try:
        version_bytes = sp.check_output(["ffmpeg", "-version"])
    except:
        return None

    # extract the version number from the output
    version_string = version_bytes.splitlines()[0].decode("utf-8")
    match = re.search(r"version\s+([\w\.]+)\s+copyright", version_string, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def is_hpc_environment() -> bool:
    """Check whether the code is running in a high-performance computing environment.

    Returns:
        bool: True if running in an HPC environment, False otherwise.
    """
    hpc_env_vars = ["SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"]
    return any(var in os.environ for var in hpc_env_vars)


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

    # check the compute environment
    result["environment"] = {"platform": sys.platform, "is_hpc": is_hpc_environment()}

    # add ffmpeg version if available
    ffmpeg_version = get_ffmpeg_version()
    if ffmpeg_version:
        result["ffmpeg version"] = ffmpeg_version

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
