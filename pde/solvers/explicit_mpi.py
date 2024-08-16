"""Defines an explicit solver using multiprocessing via MPI.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.math import OnlineStatistics
from ..tools.typing import BackendType
from .explicit import ExplicitSolver


class ExplicitMPISolver(ExplicitSolver):
    """Various explicit PDE solve using MPI.

    Warning:
        This solver can only be used if MPI is properly installed. In particular, python
        scripts then need to be started using :code:`mpirun` or :code:`mpiexec`. Please
        refer to the documentation of your MPI distribution for details.

    The main idea of the solver is to take the full initial state in the main node
    (ID 0) and split the grid into roughly equal subgrids. The main node then
    distributes these subfields to all other nodes and each node creates the right hand
    side of the PDE for itself (and independently). Each node then advances the PDE
    independently, ensuring proper coupling to neighboring nodes via special boundary
    conditions, which exchange field values between sub grids. This is implemented by
    the :meth:`get_boundary_conditions` method of the sub grids, which takes the
    boundary conditions for the full grid and creates conditions suitable for the
    specific sub grid on the given node. The trackers (and thus all input and output)
    are only handled on the main node.

    Warning:
        The function providing the right hand side of the PDE needs to support MPI. This
        is automatically the case for local evaluations (which only use the field value
        at the current position), for the differential operators provided by :mod:`pde`,
        and integration of fields. Similarly, `post_step_hook` can only be used to do
        local modifications since the field data supplied to the function is local to
        each MPI node.

    Example:
        A minimal example using the MPI solver is

        .. code-block:: python

           from pde import DiffusionPDE, ScalarField, UnitGrid

           grid = UnitGrid([64, 64])
           state = ScalarField.random_uniform(grid, 0.2, 0.3)

           eq = DiffusionPDE(diffusivity=0.1)
           result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

           if result is not None:  # restrict the output to the main node
               result.plot()

        Saving this script as `multiprocessing.py`, a parallel simulation is started by

        .. code-block:: bash

            mpiexec -n 2 python3 multiprocessing.py

        Here, the number `2` determines the number of cores that will be used. Note that
        macOS might require an additional hint on how to connect the processes even
        when they are run on the same machine (e.g., your workstation). It might help to
        run :code:`mpiexec -n 2 -host localhost python3 multiprocessing.py` in this case
    """

    name = "explicit_mpi"

    def __init__(
        self,
        pde: PDEBase,
        scheme: Literal["euler", "runge-kutta", "rk", "rk45"] = "euler",
        decomposition: Literal["auto"] | int | list[int] = "auto",
        *,
        backend: BackendType = "auto",
        adaptive: bool = False,
        tolerance: float = 1e-4,
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            scheme (str):
                Defines the explicit scheme to use. Supported values are 'euler' and
                'runge-kutta' (or 'rk' for short).
            decomposition (list of ints):
                Number of subdivision in each direction. Should be a list of length
                `grid.num_axes` specifying the number of nodes for this axis. If one
                value is `-1`, its value will be determined from the number of available
                nodes. The default value `auto` tries to determine an optimal
                decomposition by minimizing communication between nodes.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
            adaptive (bool):
                When enabled, the time step is adjusted during the simulation using the
                error tolerance set with `tolerance`.
            tolerance (float):
                The error tolerance used in adaptive time stepping. This is used in
                adaptive time stepping to choose a time step which is small enough so
                the truncation error of a single step is below `tolerance`.
        """
        pde._mpi_synchronization = self._mpi_synchronization
        super().__init__(
            pde, scheme=scheme, backend=backend, adaptive=adaptive, tolerance=tolerance
        )
        self.decomposition = decomposition

    @property
    def _mpi_synchronization(self) -> bool:  # type: ignore
        """Flag indicating whether MPI synchronization is required."""
        from ..tools import mpi

        return mpi.parallel_run

    def make_stepper(
        self, state: FieldBase, dt=None
    ) -> Callable[[FieldBase, float, float], float]:
        """Return a stepper function using an explicit scheme.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping. If `None`, this solver specifies
                1e-3 as a default value.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        from ..grids._mesh import GridMesh
        from ..tools import mpi

        if not mpi.parallel_run:
            self._logger.warning(
                "Using `ExplicitMPISolver` without a proper multiprocessing run. "
                "Scripts need to be started with `mpiexec` to profit from multiple cores"
            )

        if dt is None:
            # support `None` as a default value, so the controller can signal that
            # the solver should use a default time step
            dt = 1e-3
            if not self.adaptive:
                self._logger.warning(
                    "Explicit stepper with a fixed time step did not receive any "
                    "initial value for `dt`. Using dt=%g, but specifying a value or "
                    "enabling adaptive stepping is advisable.",
                    dt,
                )

        self.info["dt"] = dt
        self.info["dt_adaptive"] = self.adaptive
        self.info["steps"] = 0
        self.info["stochastic"] = self.pde.is_sde
        self.info["use_mpi"] = True
        self.info["scheme"] = self.scheme

        # decompose the state into multiple cells
        self.mesh = GridMesh.from_grid(state.grid, self.decomposition)
        sub_state = self.mesh.extract_subfield(state)
        self.info["grid_decomposition"] = self.mesh.shape

        if self.adaptive:
            # create stepper with adaptive steps
            self.info["dt_statistics"] = OnlineStatistics()
            adaptive_stepper = self._make_adaptive_stepper(sub_state)
            self.info["post_step_data"] = self._post_step_data_init

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """Advance `state` from `t_start` to `t_end` using adaptive steps."""
                nonlocal dt  # `dt` stores value for the next call

                # retrieve last post_step_data for this node and continue with this
                post_step_data = self.info["post_step_data"]

                # distribute the end time and the field to all nodes
                t_end = self.mesh.broadcast(t_end)
                substate_data = self.mesh.split_field_data_mpi(state.data)

                # Evolve the sub-state on each individual node. The nodes synchronize
                # field data via special boundary conditions and they synchronize the
                # maximal error via the error synchronizer. Apart from that, all nodes
                # work independently.
                t_last, dt, steps = adaptive_stepper(
                    substate_data,
                    t_start,
                    t_end,
                    dt,
                    self.info["dt_statistics"],
                    post_step_data,
                )

                # check whether dt is the same for all processes
                dt_list = self.mesh.allgather(dt)
                if not np.isclose(min(dt_list), max(dt_list)):
                    # abort simulations in all nodes when they went out of sync
                    raise RuntimeError(f"Processes went out of sync: dt={dt_list}")

                # collect the data from all nodes
                post_step_data_list = self.mesh.gather(post_step_data)
                self.mesh.combine_field_data_mpi(substate_data, out=state.data)

                if mpi.is_main:
                    self.info["steps"] += steps
                    self.info["post_step_data_list"] = post_step_data_list
                return t_last

        else:
            # create stepper with fixed steps
            fixed_stepper = self._make_fixed_stepper(sub_state, dt)
            self.info["post_step_data"] = self._post_step_data_init

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """Advance `state` from `t_start` to `t_end` using fixed steps."""
                # retrieve last post_step_data and continue with this
                post_step_data = self.info["post_step_data"]

                # calculate number of steps (which is at least 1)
                steps = max(1, int(np.ceil((t_end - t_start) / dt)))

                # distribute the number of steps and the field to all nodes
                steps = self.mesh.broadcast(steps)
                substate_data = self.mesh.split_field_data_mpi(state.data)

                # Evolve the sub-state on each individual node. The nodes synchronize
                # field data via special boundary conditions. Apart from that, all nodes
                # work independently.
                t_last = fixed_stepper(substate_data, t_start, steps, post_step_data)

                # check whether t_last is the same for all processes
                t_list = self.mesh.gather(t_last)
                if t_list is not None and not np.isclose(min(t_list), max(t_list)):
                    raise RuntimeError(f"Processes went out of sync: t_last={t_list}")

                # collect the data from all nodes
                post_step_data_list = self.mesh.gather(post_step_data)
                self.mesh.combine_field_data_mpi(substate_data, out=state.data)

                # store information in the main node
                if mpi.is_main:
                    self.info["steps"] += steps
                    self.info["post_step_data_list"] = post_step_data_list
                return t_last

        return wrapped_stepper
