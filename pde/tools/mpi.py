"""
Auxillary functions and variables for dealing with MPI multiprocessing

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
from numba.extending import overload, register_jitable

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
        """collection of operators that MPI supports"""

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
                raise AttributeError

    Operator = _OperatorRegistry()
    Operator.register("MAX", MPI.MAX)
    Operator.register("MIN", MPI.MIN)
    Operator.register("SUM", MPI.SUM)

parallel_run: bool = size > 1
"""bool: Flag indicating whether the current run is using multiprocessing"""

is_main: bool = rank == 0
"""bool: Flag indicating whether the current process is the main process (with ID 0)"""


def mpi_send(data, dest: int, tag: int) -> None:
    """send data to another MPI node

    Args:
        data: The data being send
        dest (int): The ID of the receiving node
        tag (int): A numeric tag identifying the message
    """
    MPI.COMM_WORLD.send(data, dest=dest, tag=tag)


@overload(mpi_send)
def ol_mpi_send(data, dest: int, tag: int):
    """overload the `mpi_send` function"""
    import numba_mpi

    def impl(data, dest: int, tag: int) -> None:
        """reduce a single number across all cores"""
        status = numba_mpi.send(data, dest, tag)
        assert status == 0

    return impl


def mpi_recv(data, source, tag) -> None:
    """receive data from another MPI node

    Args:
        data: A buffer into which the received data is written
        dest (int): The ID of the sending node
        tag (int): A numeric tag identifying the message

    """
    data[...] = MPI.COMM_WORLD.recv(source=source, tag=tag)


@overload(mpi_recv)
def ol_mpi_recv(data, source: int, tag: int):
    """overload the `mpi_recv` function"""
    import numba_mpi

    def impl(data, source: int, tag: int) -> None:
        """receive data from another MPI node

        Args:
            data: A buffer into which the received data is written
            dest (int): The ID of the sending node
            tag (int): A numeric tag identifying the message

        """
        status = numba_mpi.recv(data, source, tag)
        assert status == 0

    return impl


def mpi_allreduce(data, operator: int | str | None = None):
    """combines data from all MPI nodes

    Note that complex datatypes and user-defined functions are not properly supported.

    Args:
        data:
            Data being send from this node to all others
        operator:
            The operator used to combine all data. Possible options are summarized in
            the IntEnum :class:`numba_mpi.Operator`.

    Returns:
        The accumulated data
    """
    if operator:
        return MPI.COMM_WORLD.allreduce(data, op=Operator.operator(operator))
    else:
        return MPI.COMM_WORLD.allreduce(data)


@overload(mpi_allreduce)
def ol_mpi_allreduce(data, operator: int | str | None = None):
    """overload the `mpi_allreduce` function"""
    import numba_mpi

    if operator is None or isinstance(operator, nb.types.NoneType):
        op_id = -1  # value will not be used
    elif isinstance(operator, nb.types.misc.StringLiteral):
        op_id = Operator.id(operator.literal_value)
    elif isinstance(operator, nb.types.misc.Literal):
        op_id = int(operator)
    else:
        raise RuntimeError("`operator` must be a literal type")

    @register_jitable
    def _allreduce(sendobj, recvobj, operator: int | str | None = None) -> int:
        """helper function that calls `numba_mpi.allreduce`"""
        if operator is None:
            return numba_mpi.allreduce(sendobj, recvobj)  # type: ignore
        else:
            return numba_mpi.allreduce(sendobj, recvobj, op_id)  # type: ignore

    if isinstance(data, types.Number):

        def impl(data, operator: int | str | None = None):
            """reduce a single number across all cores"""
            sendobj = np.array([data])
            recvobj = np.empty((1,), sendobj.dtype)
            status = _allreduce(sendobj, recvobj, operator)
            assert status == 0
            return recvobj[0]

    elif isinstance(data, types.Array):

        def impl(data, operator: int | str | None = None):
            """reduce an array across all cores"""
            recvobj = np.empty(data.shape, data.dtype)
            status = _allreduce(data, recvobj, operator)
            assert status == 0
            return recvobj

    else:
        raise TypeError(f"Unsupported type {data.__class__.__name__}")

    return impl
