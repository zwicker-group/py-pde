"""Defines the numba backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba.extending import register_jitable

from ...grids.boundaries.local import _MPIBC, BCBase
from ..numba.backend import NumbaBackend
from ..numba.utils import jit

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...grids.base import GridBase
    from ...grids.boundaries.axis import BoundaryAxisBase
    from ...tools.typing import GhostCellSetter, NumberOrArray, NumericArray


class NumbaMPIBackend(NumbaBackend):
    """Defines MPI-compatible numba backend."""

    def _make_local_ghost_cell_setter(self, bc: BCBase) -> GhostCellSetter:
        """Return function that sets the ghost cells for a particular side of an axis.

        Args:
            bc (:class:`~pde.grids.boundaries.local.BCBase`):
                Defines the boundary conditions for a particular side, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        if not isinstance(bc, _MPIBC):
            # boundary condition is not an MPI boundary condition -> standard case
            return super()._make_local_ghost_cell_setter(bc)

        # we now deal with the MPI boundary condition
        from ...tools.mpi import mpi_recv

        cell = bc._neighbor_id
        flag = bc._mpi_flag
        num_axes = bc.grid.num_axes
        axis = bc.axis
        idx = -1 if bc.upper else 0  # index for writing data

        if num_axes == 1:

            def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                if data_full.ndim == 1:
                    # in this case, `data_full[..., idx]` is a scalar, which numba
                    # treats differently, so `numba_mpi.mpi_recv` fails
                    buffer = np.empty((), dtype=data_full.dtype)
                    mpi_recv(buffer, cell, flag)
                    data_full[..., idx] = buffer
                else:
                    mpi_recv(data_full[..., idx], cell, flag)

        elif num_axes == 2:
            if axis == 0:

                def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                    mpi_recv(data_full[..., idx, 1:-1], cell, flag)

            else:

                def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, idx], cell, flag)

        elif num_axes == 3:
            if axis == 0:

                def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                    mpi_recv(data_full[..., idx, 1:-1, 1:-1], cell, flag)

            elif axis == 1:

                def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, idx, 1:-1], cell, flag)

            else:

                def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
                    mpi_recv(data_full[..., 1:-1, 1:-1, idx], cell, flag)

        else:
            raise NotImplementedError

        return register_jitable(ghost_cell_setter)  # type: ignore

    def _make_local_ghost_cell_sender(self, bc: BCBase) -> GhostCellSetter:
        """Return function that sends data to set ghost cells for other boundaries.

        Args:
            bc (:class:`~pde.grids.boundaries.local.BCBase`):
                Defines the boundary conditions for a particular side, for which the
                sender should be defined.
        """
        if not isinstance(bc, _MPIBC):
            # boundary condition is not an MPI boundary condition -> no sending

            @register_jitable
            def noop(data_full: NumericArray, args=None) -> None:
                """No-operation as the default case."""

            return noop  # type: ignore

        # we now deal with the MPI boundary condition
        from ...tools.mpi import mpi_send

        cell = bc._neighbor_id
        flag = bc._mpi_flag
        num_axes = bc.grid.num_axes
        axis = bc.axis
        idx = -2 if bc.upper else 1  # index for reading data

        if num_axes == 1:

            def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                mpi_send(data_full[..., idx], cell, flag)

        elif num_axes == 2:
            if axis == 0:

                def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                    mpi_send(data_full[..., idx, 1:-1], cell, flag)

            else:

                def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, idx], cell, flag)

        elif num_axes == 3:
            if axis == 0:

                def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                    mpi_send(data_full[..., idx, 1:-1, 1:-1], cell, flag)

            elif axis == 1:

                def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, idx, 1:-1], cell, flag)

            else:

                def ghost_cell_sender(data_full: NumericArray, args=None) -> None:
                    mpi_send(data_full[..., 1:-1, 1:-1, idx], cell, flag)

        else:
            raise NotImplementedError

        return register_jitable(ghost_cell_sender)  # type: ignore

    def _make_axis_ghost_cell_setter(
        self, bc_axis: BoundaryAxisBase
    ) -> GhostCellSetter:
        """Return function that sets the ghost cells for a particular axis.

        Args:
            bc_axis (:class:`~pde.grids.boundaries.axis.BoundaryAxisBase`):
                Defines the boundary conditions for a particular axis, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        # get the functions that handle the data
        ghost_cell_sender_low = self._make_local_ghost_cell_sender(bc_axis.low)
        ghost_cell_sender_high = self._make_local_ghost_cell_sender(bc_axis.high)
        ghost_cell_setter_low = self._make_local_ghost_cell_setter(bc_axis.low)
        ghost_cell_setter_high = self._make_local_ghost_cell_setter(bc_axis.high)

        @register_jitable
        def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
            """Helper function setting the conditions on all axes."""
            # send boundary information to other nodes if using MPI
            ghost_cell_sender_low(data_full, args=args)
            ghost_cell_sender_high(data_full, args=args)
            # set the actual ghost cells
            ghost_cell_setter_high(data_full, args=args)
            ghost_cell_setter_low(data_full, args=args)

        return ghost_cell_setter  # type: ignore

    def make_integrator(
        self, grid: GridBase
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """

        integrate_local = self._make_local_integrator(grid)

        # deal with MPI multiprocessing
        if grid._mesh is None or len(grid._mesh) == 1:
            # standard case of a single integral
            @jit
            def integrate_global(arr: NumericArray) -> NumberOrArray:
                """Integrate data.

                Args:
                    arr (:class:`~numpy.ndarray`): discretized data on grid
                """
                return integrate_local(arr)

        else:
            # we are in a parallel run, so we need to gather the sub-integrals from
            # all subgrids in the grid mesh
            from ...tools.mpi import mpi_allreduce

            @jit
            def integrate_global(arr: NumericArray) -> NumberOrArray:
                """Integrate data over MPI parallelized grid.

                Args:
                    arr (:class:`~numpy.ndarray`): discretized data on grid
                """
                integral = integrate_local(arr)
                return mpi_allreduce(integral, operator="SUM")  # type: ignore

        return integrate_global  # type: ignore

    def make_mpi_synchronizer(
        self, operator: int | str = "MAX"
    ) -> Callable[[float], float]:
        """Return function that synchronizes values between multiple MPI processes.

        Args:
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".

        Returns:
            Function that can be used to synchronize values across nodes
        """

        from ...tools.mpi import mpi_allreduce

        @register_jitable
        def synchronize_value(error: float) -> float:
            """Return error synchronized across all cores."""
            return mpi_allreduce(error, operator=operator)  # type: ignore

        return synchronize_value  # type: ignore
