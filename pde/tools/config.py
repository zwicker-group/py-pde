"""
Handles configuration variables of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
import sys
from typing import Any, Dict, Iterator, List, Tuple, Union  # @UnusedImport

from .misc import module_available
from .parameters import Parameter


class ParameterModuleConstant:
    """ special parameter class to access module constants as configuration values """

    def __init__(self, name: str, module_path: str, variable: str):
        """
        Args:
            name (str): The name of the parameter
            module_path (str): The path of the module in which the constants is defined
            variable: The name of the constant
        """
        self.name = name
        self.module_path = module_path
        self.variable = variable

    def get(self):
        """ obtain the value of the constant """
        mod = importlib.import_module(self.module_path)
        return getattr(mod, self.variable)


# define default parameter values
DEFAULT_CONFIG: List[Union[Parameter, ParameterModuleConstant]] = [
    Parameter(
        "numba.parallel_threshold",
        256 ** 2,
        int,
        "Minimal number of support points before multithreading or multiprocessing is "
        "enabled in the numba compilations.",
    ),
    # The next items are for backward compatibility with previous mechanisms for
    # setting parameters using global constants. This fix was introduced on 2021-02-04
    # and will likely be removed around 2021-09-01.
    ParameterModuleConstant("numba.parallel", "pde.tools.numba", "NUMBA_PARALLEL"),
    ParameterModuleConstant("numba.fastmath", "pde.tools.numba", "NUMBA_FASTMATH"),
    ParameterModuleConstant("numba.debug", "pde.tools.numba", "NUMBA_DEBUG"),
]


class Config:
    """ class handling the package configuration """

    def __init__(self, mode: str = "update"):
        """
        Args:
            mode (str):
                Defines the mode in which the configuration is used. Possible values are

                * `insert`: any new configuration key can be inserted
                * `update`: only the values of pre-existing items can be updated
                * `locked`: no values can be changed
        """
        self._data: Dict[str, Any] = {p.name: p for p in DEFAULT_CONFIG}
        self.mode = mode

    def __getitem__(self, key: str):
        """ retrieve item `key` """
        parameter = self._data[key]
        if isinstance(parameter, Parameter):
            return parameter.convert()
        elif isinstance(parameter, ParameterModuleConstant):
            return parameter.get()
        else:
            return parameter

    def __setitem__(self, key: str, value):
        """ update item `key` with `value` """
        if self.mode == "insert":
            self._data[key] = value
        elif self.mode == "update":
            self[key]  # test whether the key already exist (including magic keys)
            self._data[key] = value
        elif self.mode == "locked":
            raise RuntimeError("Configuration is locked")
        else:
            raise ValueError(f"Unsupported configuration mode `{self.mode}`")

    def __iter__(self):
        return iter(self._data)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """ iterate over the items of this configuration """
        for key in self._data:
            yield key, self[key]

    def to_dict(self) -> Dict[str, Any]:
        """ convert the configuration to a simple dictionary """
        return {k: v for k, v in self.items()}


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
        """ tries to load certain python packages and returns their version """
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
        ["h5py", "pandas", "pyfftw", "tqdm"]
    )
    if module_available("numba"):
        result["numba environment"] = numba_environment()

    return result
