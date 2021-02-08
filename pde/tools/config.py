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

    def __init__(
        self, name: str, module_path: str, variable: str, description: str = ""
    ):
        """
        Args:
            name (str): The name of the parameter
            module_path (str): The path of the module in which the constants is defined
            variable: The name of the constant
        """
        self.name = name
        self.module_path = module_path
        self.variable = variable
        self.description = description

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
    ParameterModuleConstant(
        "numba.parallel",
        "pde.tools.numba",
        "NUMBA_PARALLEL",
        "Determines whether multiple cores are used in numba-compiled code.",
    ),
    ParameterModuleConstant(
        "numba.fastmath",
        "pde.tools.numba",
        "NUMBA_FASTMATH",
        "Determines whether the fastmath flag is set during compilation. This affects "
        "the precision of the mathematical calculations.",
    ),
    ParameterModuleConstant(
        "numba.debug",
        "pde.tools.numba",
        "NUMBA_DEBUG",
        "Determines whether numba used the debug mode for compilation. If enabled, "
        "this emits extra information that might be useful for debugging.",
    ),
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


def sphinx_display_config(app, what, name, obj, options, lines):
    """helper function to display default configuration in sphinx documentation

    Example:
        This function should be connected to the 'autodoc-process-docstring'
        event like so:

            app.connect('autodoc-process-docstring', sphinx_display_parameters)
    """
    if what == "class" and issubclass(obj, Parameterized):
        if any(":param parameters:" in line for line in lines):
            # parse parameters
            parameters = obj.get_parameters(sort=False)
            if parameters:
                lines.append(".. admonition::")
                lines.append(f"   Parameters of {obj.__name__}:")
                lines.append("   ")
                for p in parameters.values():
                    lines.append(f"   {p.name}")
                    text = p.description.splitlines()
                    text.append(f"(Default value: :code:`{p.default_value!r}`)")
                    text = ["     " + t for t in text]
                    lines.extend(text)
                    lines.append("")
                lines.append("")


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
