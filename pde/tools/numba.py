"""
Helper functions for just-in-time compilation with numba

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, TypeVar

import numba as nb
import numpy as np
from numba.core.types import npytypes, scalars
from numba.extending import overload, register_jitable
from numba.typed import Dict as NumbaDict  # @UnresolvedImport

from .. import config
from ..tools.misc import decorator_arguments
from .typing import Number

try:
    # is_jitted has been added in numba 0.53 on 2021-03-11
    from numba.extending import is_jitted

except ImportError:
    # for earlier version of numba, we need to define the function

    def is_jitted(function: Callable) -> bool:
        """determine whether a function has already been jitted"""
        try:
            from numba.core.dispatcher import Dispatcher
        except ImportError:
            # assume older numba module structure
            from numba.dispatcher import Dispatcher
        return isinstance(function, Dispatcher)


# numba version as a list of integers
NUMBA_VERSION = [int(v) for v in nb.__version__.split(".")[:2]]


class Counter:
    """helper class for implementing JIT_COUNT

    We cannot use a simple integer for this, since integers are immutable, so if one
    imports JIT_COUNT from this module it would always stay at the fixed value it had
    when it was first imported. The workaround would be to import the symbol every time
    the counter is read, but this is error-prone. Instead, we implement a thin wrapper
    class around an int, which only supports reading and incrementing the value. Since
    this object is now mutable it can be used easily. A disadvantage is that the object
    needs to be converted to int before it can be used in most expressions.
    """

    def __init__(self, value: int = 0):
        self._counter = value

    def __eq__(self, other):
        return self._counter == other

    def __int__(self):
        return self._counter

    def __iadd__(self, value):
        self._counter += value
        return self

    def increment(self):
        self._counter += 1

    def __repr__(self):
        return str(self._counter)


# global variable counting the number of compilations
JIT_COUNT = Counter()


TFunc = TypeVar("TFunc", bound="Callable")


def numba_environment() -> dict[str, Any]:
    """return information about the numba setup used

    Returns:
        (dict) information about the numba setup
    """
    # determine whether Nvidia Cuda is available
    try:
        from numba import cuda

        cuda_available = cuda.is_available()
    except ImportError:
        cuda_available = False

    # determine whether AMD ROC is available
    try:
        from numba import roc

        roc_available = roc.is_available()
    except ImportError:
        roc_available = False

    # determine threading layer
    try:
        threading_layer = nb.threading_layer()
    except ValueError:
        # threading layer was not initialized, so compile a mock function
        @nb.njit("i8()", parallel=True)
        def f():
            s = 0
            for i in nb.prange(4):
                s += i
            return s

        f()
        try:
            threading_layer = nb.threading_layer()
        except ValueError:  # cannot initialize threading
            threading_layer = None
    except AttributeError:  # old numba version
        threading_layer = None

    return {
        "version": nb.__version__,
        "multithreading": config["numba.multithreading"],
        "multithreading_threshold": config["numba.multithreading_threshold"],
        "fastmath": config["numba.fastmath"],
        "debug": config["numba.debug"],
        "using_svml": nb.config.USING_SVML,  # @UndefinedVariable
        "threading_layer": threading_layer,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "num_threads": nb.config.NUMBA_NUM_THREADS,  # @UndefinedVariable
        "num_threads_default": nb.config.NUMBA_DEFAULT_NUM_THREADS,  # @UndefinedVariable
        "cuda_available": cuda_available,
        "roc_available": roc_available,
    }


def flat_idx(arr: np.ndarray, i: int) -> Number:
    """helper function allowing indexing of scalars as if they arrays

    Args:
        arr
    """
    if np.isscalar(arr):
        return arr  # type: ignore
    else:
        return arr.flat[i]  # type: ignore


@overload(flat_idx)
def ol_flat_idx(arr, i):
    """helper function allowing indexing of scalars as if they arrays"""
    if isinstance(arr, nb.types.Number):
        return lambda arr, i: arr
    else:
        return lambda arr, i: arr.flat[i]


@decorator_arguments
def jit(function: TFunc, signature=None, parallel: bool = False, **kwargs) -> TFunc:
    """apply nb.jit with predefined arguments

    Args:
        function: The function which is jitted
        signature: Signature of the function to compile
        parallel (bool): Allow parallel compilation of the function
        **kwargs: Additional arguments to `nb.jit`

    Returns:
        Function that will be compiled using numba
    """
    if is_jitted(function):
        return function

    # prepare the compilation arguments
    kwargs.setdefault("nopython", True)
    if config["numba.fastmath"] is True:
        # enable some (but not all) fastmath flags. We skip the flags that affect
        # handling of infinities and NaN for safety by default. Use "fast" to enable all
        # fastmath flags; see https://llvm.org/docs/LangRef.html#fast-math-flags
        kwargs.setdefault("fastmath", {"nsz", "arcp", "contract", "afn", "reassoc"})
    else:
        kwargs.setdefault("fastmath", config["numba.fastmath"])
    kwargs.setdefault("debug", config["numba.debug"])
    # make sure parallel numba is only enabled in restricted cases
    kwargs["parallel"] = parallel and config["numba.multithreading"]

    # log some details
    logger = logging.getLogger(__name__)
    name = getattr(function, "__name__", "<anonymous function>")
    if kwargs["nopython"]:  # standard case
        logger.info("Compile `%s` with parallel=%s", name, kwargs["parallel"])
    else:  # this might imply numba falls back to object mode
        logger.warning("Compile `%s` with nopython=False", name)

    # increase the compilation counter by one
    JIT_COUNT.increment()

    return nb.jit(signature, **kwargs)(function)  # type: ignore


if nb.config.DISABLE_JIT:  # @UndefinedVariable
    # dummy function that creates a ctypes pointer
    def address_as_void_pointer(addr):
        """returns a void pointer from a given memory address

        Example:
            This can for instance be used together with `numba.carray`:

            >>> addr = arr.ctypes.data
            >>> numba.carray(address_as_void_pointer(addr), arr.shape, arr.dtype

        Args:
            addr (int): The memory address

        Returns:
            :class:`ctypes.c_void_p`: Pointer to the memory address
        """
        import ctypes

        return ctypes.cast(addr, ctypes.c_void_p)

else:
    # actually useful function that creates a numba pointer
    @nb.extending.intrinsic
    def address_as_void_pointer(typingctx, src):
        """returns a void pointer from a given memory address

        Example:
            This can for instance be used together with `numba.carray`:

            >>> addr = arr.ctypes.data
            >>> numba.carray(address_as_void_pointer(addr), arr.shape, arr.dtype

        Args:
            addr (int): The memory address

        Returns:
            :class:`numba.core.types.voidptr`: Pointer to the memory address
        """
        from numba.core import cgutils, types

        sig = types.voidptr(src)

        def codegen(cgctx, builder, sig, args):
            return builder.inttoptr(args[0], cgutils.voidptr_t)

        return sig, codegen


def make_array_constructor(arr: np.ndarray) -> Callable[[], np.ndarray]:
    """returns an array within a jitted function using basic information

    Args:
        arr (:class:`~numpy.ndarray`): The array that should be accessible within jit

    Warning:
        A reference to the array needs to be retained outside the numba code to prevent
        garbage collection from removing the array
    """

    data_addr = arr.__array_interface__["data"][0]
    strides = arr.__array_interface__["strides"]
    shape = arr.__array_interface__["shape"]
    dtype = arr.dtype

    @register_jitable
    def array_constructor() -> np.ndarray:
        """helper that reconstructs the array from the pointer and structural info"""
        data: np.ndarray = nb.carray(address_as_void_pointer(data_addr), shape, dtype)
        if strides is not None:
            data = np.lib.index_tricks.as_strided(data, shape, strides)  # type: ignore
        return data

    return array_constructor  # type: ignore


def numba_dict(data: dict[str, Any] | None = None) -> NumbaDict | None:
    """converts a python dictionary to a numba typed dictionary"""
    if data is None:
        return None
    nb_dict = NumbaDict()
    for k, v in data.items():
        nb_dict[k] = v
    return nb_dict


def get_common_numba_dtype(*args):
    r"""returns a numba numerical type in which all arrays can be represented

    Args:
        *args: All items to be tested

    Returns: numba.complex128 if any entry is complex, otherwise numba.double
    """
    for arg in args:
        if isinstance(arg, scalars.Complex):
            return nb.complex128
        elif isinstance(arg, npytypes.Array):
            if isinstance(arg.dtype, scalars.Complex):
                return nb.complex128
        else:
            raise NotImplementedError(f"Cannot handle type {arg.__class__}")
    return nb.double


@jit(nopython=True, nogil=True)
def _random_seed_compiled(seed: int) -> None:
    """sets the seed of the random number generator of numba"""
    np.random.seed(seed)


def random_seed(seed: int = 0) -> None:
    """sets the seed of the random number generator of numpy and numba

    Args:
        seed (int): Sets random seed
    """
    np.random.seed(seed)
    if not nb.config.DISABLE_JIT:  # @UndefinedVariable
        _random_seed_compiled(seed)


if NUMBA_VERSION < [0, 45]:
    warnings.warn(
        "Your numba version is outdated. Please install at least version 0.45"
    )
