"""Auxillary functions and variables for dealing with MPI multiprocessing.

Warning:
    These functions are mostly no-ops unless MPI is properly installed and python code
    was started using :code:`mpirun` or :code:`mpiexec`. Please refer to the
    documentation of your MPI distribution for details.

.. autosummary::
   :nosignatures:

   mpi_send
   mpi_recv
   mpi_allreduce

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from numba import types
from numba.extending import SentryLiteralArgs, overload, register_jitable

try:
    from numba.types import Literal
except ImportError:
    from numba.types.misc import Literal

if TYPE_CHECKING:
    from numba_mpi import Operator

# Initialize assuming that we run serial code if `numba_mpi` is not available
initialized: bool = False
"""bool: Flag determining whether mpi was initialized (and is available)"""

size: int = 1
"""int: Total process count"""

rank: int = 0
"""int: ID of the current process"""

# read state of the current MPI node
try:
    from mpi4py import MPI
except ImportError:
    # package `numba_mpi` could not be loaded
    if int(os.environ.get("PMI_SIZE", "1")) > 1:
        # environment variable indicates that we are in a parallel program
        sys.exit(
            "WARNING: Detected multiprocessing run, but could not import python "
            "package `numba_mpi`"
        )
else:
    initialized = MPI.Is_initialized()
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank

    class _OperatorRegistry:
        """Collection of operators that MPI supports."""

        _name_ids: dict[str, int]
        _ids_operators: dict[int, MPI.Op]

        def __init__(self):
            self._name_ids = {}
            self._ids_operators = {}

        def register(self, name: str, op: MPI.Op):
            op_id = int(MPI._addressof(op))
            self._name_ids[name] = op_id
            self._ids_operators[op_id] = op

        def id(self, name_or_id: int | str) -> int:
            if isinstance(name_or_id, int):
                return name_or_id
            else:
                return self._name_ids[name_or_id]

        def operator(self, name_or_id: int | str) -> MPI.Op:
            if isinstance(name_or_id, str):
                name_or_id = self._name_ids[name_or_id]
            return self._ids_operators[name_or_id]

        def __getattr__(self, name: str):
            try:
                return self._name_ids[name]
            except KeyError:
                raise AttributeError(f"MPI operator `{name}` not registered") from None

    Operator = _OperatorRegistry()
    Operator.register("MAX", MPI.MAX)
    Operator.register("MIN", MPI.MIN)
    Operator.register("SUM", MPI.SUM)

parallel_run: bool = size > 1
"""bool: Flag indicating whether the current run is using multiprocessing"""

is_main: bool = rank == 0
"""bool: Flag indicating whether the current process is the main process (with ID 0)"""


def mpi_send(data, dest: int, tag: int) -> None:
    """Send data to another MPI node.

    Args:
        data: The data being send
        dest (int): The ID of the receiving node
        tag (int): A numeric tag identifying the message
    """
    MPI.COMM_WORLD.send(data, dest=dest, tag=tag)


@overload(mpi_send)
def ol_mpi_send(data, dest: int, tag: int):
    """Overload the `mpi_send` function."""
    import numba_mpi

    def impl(data, dest: int, tag: int) -> None:
        """Reduce a single number across all cores."""
        status = numba_mpi.send(data, dest, tag)
        assert status == 0

    return impl


def mpi_recv(data, source, tag) -> None:
    """Receive data from another MPI node.

    Args:
        data: A buffer into which the received data is written
        dest (int): The ID of the sending node
        tag (int): A numeric tag identifying the message
    """
    data[...] = MPI.COMM_WORLD.recv(source=source, tag=tag)


@overload(mpi_recv)
def ol_mpi_recv(data, source: int, tag: int):
    """Overload the `mpi_recv` function."""
    import numba_mpi

    def impl(data, source: int, tag: int) -> None:
        """Receive data from another MPI node.

        Args:
            data: A buffer into which the received data is written
            dest (int): The ID of the sending node
            tag (int): A numeric tag identifying the message
        """
        status = numba_mpi.recv(data, source, tag)
        assert status == 0

    return impl


def mpi_allreduce(data, operator):
    """Combines data from all MPI nodes.

    Note that complex datatypes and user-defined reduction operators are not properly
    supported in numba-compiled cases.

    Args:
        data:
            Data being send from this node to all others
        operator:
            The operator used to combine all data. Possible options are summarized in
            the IntEnum :class:`numba_mpi.Operator`.

    Returns:
        The accumulated data
    """
    if not parallel_run:
        # in a serial run, we can always return the value as is
        return data

    if isinstance(data, np.ndarray):
        # synchronize an array
        out = np.empty_like(data)
        MPI.COMM_WORLD.Allreduce(data, out, op=Operator.operator(operator))
        return out

    else:
        # synchronize a single value
        return MPI.COMM_WORLD.allreduce(data, op=Operator.operator(operator))


@overload(mpi_allreduce)
def ol_mpi_allreduce(data, operator):
    """Overload the `mpi_allreduce` function."""
    if size == 1:
        # We can simply return the value in a serial run

        def impl(data, operator):
            return data

        return impl

    # Conversely, in a parallel run, we need to use the correct reduction. Let's first
    # determine the operator, which must be given as a literal type
    SentryLiteralArgs(["operator"]).for_function(ol_mpi_allreduce).bind(data, operator)
    if isinstance(operator, Literal):
        # an operator is specified (using a literal value)
        if isinstance(operator.literal_value, str):
            # an operator is specified by it's name
            op_id = Operator.id(operator.literal_value)
        else:
            # assume an operator is specified by it's id
            op_id = int(operator.literal_value)
    elif isinstance(operator, nb.types.Integer):
        op_id = None  # use given value of operator
    else:
        raise RuntimeError(f"`operator` must be a literal type, not {operator}")

    import numba_mpi

    @register_jitable
    def _allreduce(sendobj, recvobj, operator) -> int:
        """Helper function that calls `numba_mpi.allreduce`"""
        if op_id is None:
            return numba_mpi.allreduce(sendobj, recvobj, operator)  # type: ignore
        else:
            return numba_mpi.allreduce(sendobj, recvobj, op_id)  # type: ignore

    if isinstance(data, types.Number):
        # implementation of the reduction for a single number

        def impl(data, operator):
            """Reduce a single number across all cores."""
            sendobj = np.array([data])
            recvobj = np.empty((1,), sendobj.dtype)
            status = _allreduce(sendobj, recvobj, operator)
            if status != 0:
                raise RuntimeError
            return recvobj[0]

    elif isinstance(data, types.Array):
        # implementation of the reduction for a numpy array

        def impl(data, operator):
            """Reduce an array across all cores."""
            recvobj = np.empty(data.shape, data.dtype)
            status = _allreduce(data, recvobj, operator)
            if status != 0:
                raise RuntimeError
            return recvobj

    else:
        raise TypeError(f"Unsupported type {data.__class__.__name__}")

    return impl
