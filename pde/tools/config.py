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

import contextlib
import copy
import importlib
import os
import re
import subprocess as sp
import sys
import warnings
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union

from typing_extensions import Self

from .misc import module_available
from .nested_dict import NestedDict

ParameterTypes = str | float | int | bool | None


class _DEFAULT_TYPE:
    """Sentinel type indicating that the default value should be inferred."""


_DEFAULT = _DEFAULT_TYPE()  # signals that the default value should be used


class _OMITTED_TYPE:
    """Sentinel type indicating that an argument was not supplied."""


_OMITTED = _OMITTED_TYPE()  # signals that the default value should be used


@dataclass
class Parameter:
    """Class representing a single parameter.

    Args:
        value:
            The current value of the parameter.
        default_value:
            The fallback value used when the parameter is reset. If omitted, the
            current value is used.
        cls:
            Type used to convert assigned values.
        description:
            Human-readable explanation of the parameter.
        hidden:
            Flag indicating whether the parameter should be hidden in summaries.
        extra:
            Optional metadata stored alongside the parameter.
    """

    value: ParameterTypes
    _: KW_ONLY
    default_value: ParameterTypes | _OMITTED_TYPE = _OMITTED
    cls: object | _OMITTED_TYPE = _OMITTED
    description: str = ""
    hidden: bool = False
    extra: dict[str, Any] | None = None

    def __post_init__(self):
        """Normalize omitted defaults and inferred conversion types.

        The method fills in omitted constructor arguments and ensures that
        :attr:`extra` is always a dictionary.
        """
        # determine a suitable default value for the parameter
        if self.default_value is _OMITTED:
            self.default_value = None if self.value is _DEFAULT else self.value
        if self.cls is _OMITTED:
            if self.default_value is None:
                self.cls = object
            elif isinstance(self.default_value, int):
                self.cls = float
            else:
                self.cls = type(self.default_value)
        # determine a suitable type for the parameter
        if self.extra is None:
            self.extra = {}
        #     if extra is None else extra

    def convert(
        self, value: ParameterTypes | _OMITTED_TYPE = _OMITTED
    ) -> ParameterTypes:
        """Converts a `value` into the correct type for this parameter. If `value` is
        not given, the current value is converted.

        Note that this does not make a copy of the values, which could lead to
        unexpected effects where the default value is changed by an instance.

        Args:
            value: The value to convert

        Returns:
            The converted value, which is of type `self.cls`
        """
        if value is _OMITTED:
            value = self.value
        assert not isinstance(value, _OMITTED_TYPE)

        if self.cls is _OMITTED:
            if self.default_value is None:
                self.cls = object
            elif isinstance(self.default_value, int):
                self.cls = float  # assume that numbers can in principle be floats
            else:
                self.cls = type(self.value)

        if self.cls is object:
            return value
        try:
            return self.cls(value)  # type: ignore
        except ValueError as err:
            msg = f"Could not convert {value!r} to {self.cls!r} for parameter."
            raise ValueError(msg) from err

    def reset(self) -> None:
        """Reset parameter to default value."""
        if isinstance(self.default_value, _OMITTED_TYPE):
            msg = "Default value is not set"
            raise RuntimeError(msg)  # noqa: TRY004
        self.value = self.default_value


ConfigValue = Union["Config", Parameter]
ConfigLike = Union[Sequence[Parameter], MutableMapping[str, Any], "Config"]


class Modes(Enum):
    """Access modes controlling how configuration entries can be modified."""

    INSERT = "insert"
    UPDATE = "update"
    READONLY = "readonly"


@dataclass
class ConfigMode:
    """Mutable object storing the current configuration mode.

    Args:
        mode:
            Initial mode controlling whether items can be inserted, updated, or
            modified at all.
    """

    node: Modes = Modes.UPDATE
    leaf: Modes = Modes.INSERT
    delete: bool = False

    @classmethod
    def from_str(cls, value: str):
        """Create a mode descriptor from a textual mode name.

        Args:
            value:
                Mode name. Supported values are ``"insert"``, ``"update"``, and
                ``"locked"``.

        Returns:
            ConfigMode:
                Newly created mode descriptor.

        Raises:
            ValueError:
                If `value` is not one of the supported mode names.
        """
        if value == "update":
            return cls(node=Modes.UPDATE, leaf=Modes.UPDATE)
        if value == "insert":
            return cls(node=Modes.INSERT, leaf=Modes.INSERT)
        if value == "readonly":
            return cls(node=Modes.READONLY, leaf=Modes.READONLY)
        raise ValueError(value)

    def _getstate(self):
        """Return a serializable snapshot of the current mode settings."""
        return self.__dict__.copy()

    def _setstate(
        self,
        node: Modes | None = None,
        leaf: Modes | None = None,
        delete: bool | None = None,
    ):
        """Update selected mode flags in place.

        Args:
            node:
                Optional replacement for the node-update mode.
            leaf:
                Optional replacement for the leaf-update mode.
            delete:
                Optional replacement for the deletion-permission flag.
        """
        if node is not None:
            self.node = Modes(node)
        if leaf is not None:
            self.leaf = Modes(leaf)
        if delete is not None:
            self.delete = delete


class AccessError(RuntimeError):
    """Raised when a configuration change violates the active access mode."""


class _ConfigDataDict(dict):
    """Dictionary enforcing write-access restrictions through a shared mode object."""

    _mode: ConfigMode

    def __init__(self, mode: ConfigMode):
        """Initialize the lock-aware dictionary.

        Args:
            mode:
                Shared mode descriptor controlling which write operations are allowed.
        """
        self._mode = mode

    def _allow(self, cat: Literal["node", "leaf"], modes: Iterable[Modes]) -> bool:
        """Check whether a write category is permitted under the current mode.

        Args:
            cat:
                Category to validate. Must be either ``"node"`` or ``"leaf"``.
            modes:
                Set of allowed :class:`Modes` values.

        Returns:
            bool:
                `True` if the category is currently in one of the allowed modes.

        Raises:
            ValueError:
                If `cat` is not a recognized category.
        """
        if cat == "node":
            mode = self._mode.node
        elif cat == "leaf":
            mode = self._mode.leaf
        else:
            raise ValueError
        return mode in modes

    def _convert_value(self, value: Any) -> Parameter | Config:
        """Convert user input to a valid configuration value.

        Args:
            value:
                Value supplied for a configuration entry.

        Returns:
            Parameter | Config:
                Normalized configuration object representing `value`.
        """
        if isinstance(value, (Parameter, Config)):
            return value
        if isinstance(value, dict):
            return Config(value, mode=self._mode)

        if value is None:
            cls = object
        elif isinstance(value, int):
            cls = float  # assume that numbers can in principle be floats
        else:
            cls = type(value)
        return Parameter(default_value=value, cls=cls, value=value)

    def __getitem__(self, key: str) -> Config | Parameter:
        """Returns an item using dictionary indexing syntax.

        Args:
            key (str):
                Key or nested key path to resolve.

        Returns:
            Any:
                Value associated with `key`.
        """
        value = super().__getitem__(key)
        if not isinstance(value, (Config, Parameter)):
            raise TypeError
        return value

    def __setitem__(self, key, value):
        """Set an item while validating node/leaf write permissions.

        Args:
            key:
                Item key to set.
            value:
                Value to write at `key`.

        Raises:
            AccessError:
                If the active mode forbids the attempted insertion/update.
        """
        if isinstance(value, MutableMapping):
            # deal with node
            if key in self:
                msg = "Cannot update whole subtrees"
                raise RuntimeError(msg)
            # insert node
            if not self._allow("node", {Modes.INSERT}):
                msg = f"Dictionary does not permit adding items for mode {self}"
                raise AccessError(msg)
            super().__setitem__(key, self._convert_value(value))

        # deal with leaf item
        elif key in self:
            # update leaf
            if not self._allow("leaf", {Modes.UPDATE, Modes.INSERT}):
                msg = f"Dictionary is locked and cannot be updated for mode {self}"
                raise AccessError(msg)
            if isinstance(value, Parameter):
                super().__setitem__(key, self._convert_value(value))
            else:
                super().__getitem__(key).value = value
        else:
            # insert leaf
            if not self._allow("leaf", {Modes.INSERT}):
                msg = f"Dictionary does not permit adding items for mode {self}"
                raise AccessError(msg)
            super().__setitem__(key, self._convert_value(value))

    def __delitem__(self, key):
        """Delete an item if deletions are enabled by the active mode.

        Args:
            key:
                Item key to delete.

        Raises:
            AccessError:
                If the active mode forbids deletions.
        """
        if hasattr(self, "_mode") and not self._mode.delete:
            msg = "Cannot delete items from locked dictionary for mode {self}"
            raise AccessError(msg)
        super().__delitem__(key)

    def clear(self):
        """Delete all items if deletions are enabled by the active mode.

        Args:
            key:
                Item key to delete.

        Raises:
            AccessError:
                If the active mode forbids deletions.
        """
        if hasattr(self, "_mode") and not self._mode.delete:
            msg = "Cannot delete items from locked dictionary for mode {self}"
            raise AccessError(msg)
        super().clear()


class Config(NestedDict[Parameter]):
    """Class handling general (nested) configurations.

    Configurations are basically (nested) dictionaries with string keys that hold
    :class:`Parameter` values, which contain a value with some extra information.
    Moreover, configurations have a `mode` that controls whether the configuration is
    writeable or not.
    """

    _mode: ConfigMode

    def __init__(
        self, items: ConfigLike | None = None, *, mode: ConfigMode | str = "update"
    ):
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
        if isinstance(mode, ConfigMode):
            self._mode = mode  # inherit mode from parent
        elif isinstance(mode, str):
            self._mode = ConfigMode.from_str(mode)  # generate initial mode object
        else:
            raise TypeError

        # initialize empty configuration
        super().__init__()

        if items:
            # temporarily allow inserting items to add items
            with self.changed_mode(node="insert", leaf="insert"):
                if isinstance(items, Config):
                    self.replace_recursive(
                        items.to_dict(values=False), delete_extra=False
                    )
                elif isinstance(items, MutableMapping):
                    self.replace_recursive(items, delete_extra=False)
                else:
                    raise TypeError

    def _make_dict(self):
        """Create the backing dictionary enforcing the current config mode."""
        return _ConfigDataDict(mode=self.mode)

    def _make_node(self) -> Self:
        """Create a child configuration node inheriting the current mode."""
        return self.__class__(mode=self.mode)

    @property
    def mode(self) -> ConfigMode:
        """Current mutable mode descriptor shared across the whole config tree."""
        return self._mode

    @mode.setter
    def mode(self, mode: ConfigMode) -> None:
        """Update the active configuration mode.

        Args:
            mode:
                New mode controlling whether inserts, updates, or deletions are
                permitted.
        """
        # keep the identify and rather update the values
        self._mode._setstate(**mode._getstate())

    @contextlib.contextmanager
    def changed_mode(self, **kwargs):
        """Temporarily switch to `mode` and restore the previous mode afterwards.

        Args:
            **kwargs:
                Keyword arguments forwarded to :meth:`ConfigMode._setstate`, such as
                `node`, `leaf`, and `delete`.

        Yields:
            ConfigMode:
                The mode controller with the temporary mode applied.
        """
        old_state = self.mode._getstate()
        self.mode._setstate(**kwargs)
        try:
            yield self
        finally:
            self.mode._setstate(**old_state)

    def _get_raw_item(self, key: str) -> Any:
        """Retrieve an item without converting `Parameter` instances."""
        return NestedDict.__getitem__(self, key)

    def __getitem__(self, key: str):
        """Retrieve item `key`.

        Args:
            key (str): The configuration key
        """
        value = NestedDict.__getitem__(self, key)
        if isinstance(value, NestedDict):
            return value
        if isinstance(value, Parameter):
            return value.convert()
        raise TypeError(value)

    def replace_recursive(
        self, other: MutableMapping[str, Any], delete_extra: bool = False
    ) -> None:
        """Recursively replaces data of the current instance by another mapping.

        Args:
            other (MutableMapping[str, Any]):
                Mapping whose entries are will end up in this object.
        """
        if not isinstance(other, MutableMapping):
            raise TypeError

        # update all values from `other`
        seen: set[str] = set()
        for k, v in other.items():
            if isinstance(v, MutableMapping):
                new_node = self.create_node(k)
                assert isinstance(new_node, Config)
                new_node.replace_recursive(v, delete_extra=delete_extra)
            elif k in self and not isinstance(v, Parameter):
                self._get_raw_item(k).value = v
            else:
                self[k] = v
            seen.add(k)

        # delete all items that were not in the other mapping
        if delete_extra:
            for k in set(self.keys()) - seen:
                del self[k]

    def to_dict(self, flatten: bool = False, values: bool = False) -> dict[str, Any]:
        """Convert the configuration to a simple dictionary.

        Args:
            flatten (bool):
                Return flat or nested dictionary.
            values (bool):
                Whether to return only values (and not :class:`Parameter` instances)

        Returns:
            dict: A representation of the configuration in a normal :class:`dict`.
        """
        if flatten:
            res = dict(NestedDict.items(self, flatten=True))
            if values:
                return {k: v.value for k, v in res.items()}
            return res
        # return hierarchical dictionaries
        return {
            k: v.to_dict(values=values, flatten=False)
            if isinstance(v, Config)
            else (v.value if values else v)  # type: ignore
            for k, v in self.data.items()
        }

    def __repr__(self) -> str:
        """Represent the configuration as a string."""
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    def copy(self) -> Config:
        """Creates a structural copy with copied nested mappings.

        Child dictionaries and child `NestedDict` instances are copied, while
        non-mapping leaf values are reused by reference.

        Returns:
            NestedDict:
                New instance containing copied nested structure.
        """
        data: dict[str, Any] = {}
        for key, value in self.data.items():
            if isinstance(value, Config):
                data[key] = value.copy()
            elif isinstance(value, Parameter):
                data[key] = copy.copy(value)
            else:
                data[key] = value
        return self.__class__(data, mode=self.mode)

    @contextlib.contextmanager
    def __call__(self, values: dict[str, Any] | None = None, **kwargs):
        """Context manager temporarily changing the configuration.

        Args:
            values (dict):
                Mapping with temporary configuration values.
            **kwargs:
                Additional temporary configuration values.

        Yields:
            None:
                Control returns to the caller while the temporary configuration is
                active.
        """
        old_data = self.to_dict(values=True)  # save old values
        # update configuration with new values
        if values is not None:
            self.replace_recursive(values, delete_extra=False)
        self.replace_recursive(kwargs, delete_extra=False)

        # return to caller to have the updated configuration
        try:
            yield
        finally:
            # restore old configuration
            with self.changed_mode(node="insert", leaf="insert", delete=True):
                self.replace_recursive(old_data, delete_extra=True)


config = Config({"backend": {}}, mode="update")

with config.changed_mode(node="insert", leaf="insert"):
    # define default parameter values
    config["operators.conservative_stencil"] = Parameter(
        value=True,
        cls=bool,
        description="Indicates whether conservative stencils should be used for "
        "differential operators on curvilinear grids. Conservative operators ensure "
        "mass conservation at slightly slower computation speed. Note that some "
        "backends might ignore this option.",
    )
    config["operators.tensor_symmetry_check"] = Parameter(
        value=True,
        cls=bool,
        description="Indicates whether tensor fields are checked for having a suitable "
        "form for evaluating differential operators in curvilinear coordinates where "
        "some axes are assumed to be symmetric. In such cases, some tensor components "
        "might need to vanish, so the result of the operator can be expressed. Note "
        "that some backends might ignore this option.",
    )
    config["operators.cartesian.laplacian_2d_corner_weight"] = Parameter(
        value=0.0,
        cls=float,
        description="Weighting factor for the corner points of the 2d cartesian "
        "Laplacian stencil. The standard value is zero, corresponding to the "
        "traditional 5-point stencil. Alternative choices are 1/2 (Oono-Puri stencil) "
        "and 1/3 (Patra-Karttunen or Mehrstellen stencil); see "
        "https://en.wikipedia.org/wiki/Nine-point_stencil. Note that some backends "
        "might ignore this option.",
    )
    config["boundaries.accept_lists"] = Parameter(
        value=True,
        cls=bool,
        description="Indicate whether boundary conditions can be set using the "
        "deprecated legacy format, where conditions for individual axes and sides "
        "where set using lists. If disabled, only the new format using dicts is "
        "supported.",
    )
    config["default_backend"] = Parameter(
        value="numba",
        cls=str,
        description="Indicate which backend is selected by default.",
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
    """Helper function converting a version string into a list of integers.

    Args:
        ver_str (str): The version string to parse

    Returns:
        list[int]: List of version numbers as integers
    """
    result = []
    for token in ver_str.split(".")[:3]:
        with contextlib.suppress(ValueError):
            result.append(int(token))
    return result


def check_package_version(package_name: str, min_version: str):
    """Checks whether a package has a sufficient version.

    Args:
        package_name (str): The name of the package to check
        min_version (str): The minimum required version

    Returns:
        None: The function only emits warnings and does not return a value.
    """

    msg = f"`{package_name}` version {min_version} required for py-pde"
    try:
        # obtain version of the package
        version = importlib.import_module(package_name).__version__

    except ImportError:
        warnings.warn(f"{msg} (but none installed)", stacklevel=2)

    else:
        # check whether it is installed and works
        if parse_version_str(version) < parse_version_str(min_version):
            warnings.warn(f"{msg} (installed: {version})", stacklevel=2)


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
    """Read version number of ffmpeg program.

    Returns:
        str | None:
            Detected version string, or `None` if ffmpeg is unavailable or the
            version could not be parsed.
    """
    # run ffmpeg to get its version
    try:
        version_bytes = sp.check_output(["ffmpeg", "-version"])
    except Exception:
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

    from pde import config

    from .. import __version__ as package_version
    from ..backends.numba.utils import numba_environment
    from . import mpi
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
    result["config"] = config.to_dict(flatten=True, values=True)

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

    backend = {}
    if module_available("numba"):
        backend["numba"] = numba_environment()
    if backend:
        result["backend"] = backend

    # add information about MPI environment
    if mpi.initialized:
        result["multiprocessing"] = {"initialized": True, "size": mpi.size}
    else:
        result["multiprocessing"] = {"initialized": False}

    return result
