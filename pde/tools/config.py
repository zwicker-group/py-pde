"""
Module defining a class for handling the package configuration.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Dict  # @UnusedImport


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
        self._data: Dict[str, Any] = {}
        self.mode = mode

    def __getitem__(self, key: str):
        """ retrieve item `key` """
        try:
            # try reading the information from the internal dictionary
            return self._data[key]

        except KeyError:
            # if this didn't work, import some magic constants
            # This is for backward compatibility (introduced 2021-02-04)
            if key == "numba.parallel":
                from .numba import NUMBA_PARALLEL

                return NUMBA_PARALLEL
            elif key == "numba.fastmath":
                from .numba import NUMBA_FASTMATH

                return NUMBA_FASTMATH
            elif key == "numba.debug":
                from .numba import NUMBA_DEBUG

                return NUMBA_DEBUG
            else:
                raise  # raise KeyError for unknown variables

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
