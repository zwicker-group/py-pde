"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


class Config:
    """ class handling the package configuration """

    def __init__(self, locked: bool = False):
        self._data = {}
        self.locked = locked

    def __getitem__(self, key: str):
        """ retrieve item """
        try:
            return self._data[key]
        except KeyError:
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
                raise

    def __setitem__(self, key: str, value):
        if self.locked:
            raise RuntimeError("Configuration is locked")
        else:
            self._data[key] = value
