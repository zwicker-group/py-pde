"""
Handles configuration variables of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import importlib
from typing import Any, Dict, List, Union  # @UnusedImport

from .parameters import Parameter


class ParameterModuleConstant:
    """ special parameter class to access module values """

    def __init__(self, name: str, module_path: str, variable: str):
        self.name = name
        self.module_path = module_path
        self.variable = variable

    def get(self):
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
                Defines the mode in which the configuration is used. If `insert`, any
                new configuration key can be inserted. If `update`, only the values of
                pre-existing items can be updated. If `locked`, no values can be
                changed.
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
