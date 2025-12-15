"""Defines utilities for the numba backend.

.. autosummary::
   :nosignatures:

   make_get_valid
   make_get_arr_1d
   numba_environment
   jit
   make_array_constructor
   numba_dict
   get_common_numba_dtype
   random_seed

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

import numba as nb
import numpy as np
from numba.core.types import npytypes, scalars
from numba.extending import is_jitted, overload, register_jitable
from numba.typed import Dict as NumbaDict
from typing_extensions import Self

from ... import config
from ...tools.misc import decorator_arguments
from ...tools.typing import NumericArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...tools.typing import Number


if TYPE_CHECKING:
    from ...grids.base import GridBase

# numba version as a list of integers
NUMBA_VERSION = [int(v) for v in nb.__version__.split(".")[:2]]


def make_get_valid(grid: GridBase) -> Callable[[NumericArray], NumericArray]:
    """Create a function to extract the valid part of a full data array.

    Returns:
        callable: Mapping a numpy array containing the full data of the grid to a
            numpy array of only the valid data
    """
    num_axes = grid.num_axes

    @register_jitable
    def get_valid(data_full: NumericArray) -> NumericArray:
        """Return valid part of the data (without ghost cells)

        Args:
            data_full (:class:`~numpy.ndarray`):
                The array with ghost cells from which the valid data is extracted
        """
        if num_axes == 1:
            return data_full[..., 1:-1]
        if num_axes == 2:
            return data_full[..., 1:-1, 1:-1]
        if num_axes == 3:
            return data_full[..., 1:-1, 1:-1, 1:-1]
        raise NotImplementedError

    return get_valid  # type: ignore


def make_get_arr_1d(
    dim: int, axis: int
) -> Callable[[NumericArray, tuple[int, ...]], tuple[NumericArray, int, tuple]]:
    """Create function that extracts a 1d array at a given position.

    Args:
        dim (int):
            The dimension of the space, i.e., the number of axes in the supplied array
        axis (int):
            The axis that is returned as the 1d array

    Returns:
        function: A numba compiled function that takes the full array `arr` and
        an index `idx` (a tuple of `dim` integers) specifying the point where
        the 1d array is extract. The function returns a tuple (arr_1d, i, bc_i),
        where `arr_1d` is the 1d array, `i` is the index `i` into this array
        marking the current point and `bc_i` are the remaining components of
        `idx`, which locate the point in the orthogonal directions.
        Consequently, `i = idx[axis]` and `arr[..., idx] == arr_1d[..., i]`.
    """
    assert 0 <= axis < dim
    ResultType = tuple[NumericArray, int, tuple]

    # extract the correct indices
    if dim == 1:

        def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
            """Extract the 1d array along axis at point idx."""
            i = idx[0]
            bc_idx: tuple = (...,)
            arr_1d = arr
            return arr_1d, i, bc_idx

    elif dim == 2:
        if axis == 0:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y = idx
                bc_idx = (..., y)
                arr_1d = arr[..., :, y]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i = idx
                bc_idx = (..., x)
                arr_1d = arr[..., x, :]
                return arr_1d, i, bc_idx

    elif dim == 3:
        if axis == 0:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                i, y, z = idx
                bc_idx = (..., y, z)
                arr_1d = arr[..., :, y, z]
                return arr_1d, i, bc_idx

        elif axis == 1:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, i, z = idx
                bc_idx = (..., x, z)
                arr_1d = arr[..., x, :, z]
                return arr_1d, i, bc_idx

        elif axis == 2:

            def get_arr_1d(arr: NumericArray, idx: tuple[int, ...]) -> ResultType:
                """Extract the 1d array along axis at point idx."""
                x, y, i = idx
                bc_idx = (..., x, y)
                arr_1d = arr[..., x, y, :]
                return arr_1d, i, bc_idx

    else:
        raise NotImplementedError

    return register_jitable(inline="always")(get_arr_1d)  # type: ignore


class Counter:
    """Helper class for implementing JIT_COUNT.

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

    def __iadd__(self, value) -> Self:
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
    """Return information about the numba setup used.

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
        "using_svml": nb.config.USING_SVML,
        "threading_layer": threading_layer,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "num_threads": nb.config.NUMBA_NUM_THREADS,
        "num_threads_default": nb.config.NUMBA_DEFAULT_NUM_THREADS,
        "cuda_available": cuda_available,
        "roc_available": roc_available,
    }


def flat_idx(arr: NumericArray, i: int) -> Number:
    """Helper function allowing indexing of scalars as if they arrays.

    Args:
        arr
    """
    if np.isscalar(arr):
        return arr  # type: ignore
    return arr.flat[i]  # type: ignore


@overload(flat_idx)
def ol_flat_idx(arr, i):
    """Helper function allowing indexing of scalars as if they arrays."""
    if isinstance(arr, nb.types.Number):
        return lambda arr, i: arr
    return lambda arr, i: arr.flat[i]


@decorator_arguments
def jit(function: TFunc, signature=None, parallel: bool = False, **kwargs) -> TFunc:
    """Apply nb.jit with predefined arguments.

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
    if config["numba.fastmath"] is True:
        # enable some (but not all) fastmath flags. We skip the flags that affect
        # handling of infinities and NaN for safety by default. Use "fast" to enable all
        # fastmath flags; see https://llvm.org/docs/LangRef.html#fast-math-flags
        kwargs.setdefault("fastmath", {"nsz", "arcp", "contract", "afn", "reassoc"})
    else:
        kwargs.setdefault("fastmath", config["numba.fastmath"])
    kwargs.setdefault("debug", config["numba.debug"])
    # make sure parallel numba is only enabled in restricted cases
    kwargs["parallel"] = parallel and config.use_multithreading()

    # log some details
    logger = logging.getLogger(__name__)
    name = getattr(function, "__name__", "<anonymous function>")
    if kwargs["parallel"]:
        logger.info("Compile `%s`", name)
    else:
        logger.info("Compile `%s` with parallel=True", name)

    # increase the compilation counter by one
    JIT_COUNT.increment()

    return nb.jit(signature, **kwargs)(function)  # type: ignore


if nb.config.DISABLE_JIT:
    # dummy function that creates a ctypes pointer
    def address_as_void_pointer(addr):
        """Returns a void pointer from a given memory address.

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
        """Returns a void pointer from a given memory address.

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


def make_array_constructor(arr: NumericArray) -> Callable[[], NumericArray]:
    """Returns an array within a jitted function using basic information.

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
    def array_constructor() -> NumericArray:
        """Helper that reconstructs the array from the pointer and structural info."""
        data: NumericArray = nb.carray(address_as_void_pointer(data_addr), shape, dtype)
        if strides is not None:
            data = np.lib.stride_tricks.as_strided(data, shape, strides)
        return data

    return array_constructor  # type: ignore


def numba_dict(data: dict[str, Any] | None = None, /, **kwargs) -> NumbaDict:
    """Converts a python dictionary to a numba typed dictionary.

    Args:
        data (dict):
            Data to be converted to a dictionary
        **kwargs:
            Additional items added to the dictionary

    Returns:
        :class:`~numba.typed.Dict`: A dictionary of numba type
    """
    nb_dict = NumbaDict()
    if data is not None:
        for k, v in data.items():
            nb_dict[k] = v
    for k, v in kwargs.items():
        nb_dict[k] = v
    return nb_dict


def get_common_numba_dtype(*args):
    r"""Returns a numba numerical type in which all arrays can be represented.

    Args:
        *args: All items to be tested

    Returns: numba.complex128 if any entry is complex, otherwise numba.double
    """
    for arg in args:
        if isinstance(arg, scalars.Complex):
            return nb.complex128
        if isinstance(arg, npytypes.Array):
            if isinstance(arg.dtype, scalars.Complex):
                return nb.complex128
        else:
            msg = f"Cannot handle type {arg.__class__}"
            raise NotImplementedError(msg)
    return nb.double


@jit(nogil=True)
def _random_seed_compiled(seed: int) -> None:
    """Sets the seed of the random number generator of numba."""
    np.random.seed(seed)  # noqa: NPY002


def random_seed(seed: int = 0) -> None:
    """Sets the seed of the random number generator of numpy and numba.

    Args:
        seed (int): Sets random seed
    """
    np.random.seed(seed)  # noqa: NPY002
    if not nb.config.DISABLE_JIT:
        _random_seed_compiled(seed)


if NUMBA_VERSION < [0, 59]:
    warnings.warn(
        "Your numba version is outdated. Please install at least version 0.59",
        stacklevel=2,
    )
