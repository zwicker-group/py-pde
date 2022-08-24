"""
Defines an explicit solver using multiprocessing via MPI
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, List, Tuple, Union

import numba_mpi
import numpy as np

from ..fields.base import FieldBase
from ..grids.mesh import GridMesh
from ..pdes.base import PDEBase
from ..tools import mpi
from ..tools.numba import jit
from .base import SolverBase


class ExplicitMPISolver(SolverBase):
    """class for solving partial differential equations explicitly using MPI

    This solver can only be used if MPI is properly installed.

    Warning:
        `modify_after_step` can only be used to do local modifications since the field
        data supplied to the function is local to each MPI node.
    """

    name = "explicit_mpi"

    dt_min = 1e-10
    dt_max = 1e10

    def __init__(
        self,
        pde: PDEBase,
        decomposition: Union[int, List[int]] = -1,
        *,
        backend: str = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The instance describing the pde that needs to be solved
            decomposition (list of ints):
                Number of subdivision in each direction. Should be a list of length
                `grid.num_axes` specifying the number of nodes for this axis. If one
                value is `-1`, its value will be determined from the number of available
                nodes. The default value decomposed the first axis using all available
                nodes.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
        """
        super().__init__(pde)
        self.backend = backend
        self.decomposition = decomposition

    def _make_fixed_euler_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int], Tuple[float, float]]:
        """make a simple Euler stepper with fixed time step

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, steps: int)`
        """
        self.info["dt_adaptive"] = False

        if self.pde.is_sde:
            raise NotImplementedError

        # extract data frim subgrid
        sub_state = self.mesh.extract_subfield(state)

        # obtain function of the PDE
        rhs_pde = self._make_pde_rhs(sub_state, backend=self.backend)
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        def stepper(
            sub_state_data: np.ndarray, t_start: float, steps: int
        ) -> Tuple[float, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt
                sub_state_data += dt * rhs_pde(sub_state_data, t)
                modifications += modify_after_step(sub_state_data)

            return t + dt, modifications

        self.info["stochastic"] = False
        self._logger.info(f"Initialized explicit MPI-Euler stepper with dt=%g", dt)

        return stepper

    def make_stepper(
        self, state: FieldBase, dt=None
    ) -> Callable[[FieldBase, float, float], float]:
        """return a stepper function using an explicit scheme

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
        # support `None` as a default value, so the controller can signal that
        # the solver should use a default time step
        if dt is None:
            dt = 1e-3
            if not self.adaptive:
                self._logger.warning(
                    "Explicit stepper with a fixed time step did not receive any "
                    f"initial value for `dt`. Using dt={dt}, but specifying a value or "
                    "enabling adaptive stepping is advisable."
                )

        self.info["dt"] = dt
        self.info["steps"] = 0
        self.info["state_modifications"] = 0.0
        self.info["use_mpi"] = True
        # self.info["scheme"] = self.scheme

        # decompose the state into multiple cells
        self.mesh = GridMesh.from_grid(state.grid, self.decomposition)
        self.info["grid_decomposition"] = self.mesh.shape

        if self.pde.is_sde and self.adaptive:
            self._logger.warning(
                "Adaptive stochastic stepping is not implemented. Using a fixed time "
                "step instead."
            )

        # create stepper with fixed steps
        # if self.scheme == "euler":
        fixed_stepper = self._make_fixed_euler_stepper(state, dt)
        # elif self.scheme in {"runge-kutta", "rk", "rk45"}:
        #     fixed_stepper = self._make_rk45_stepper(state, dt)
        # else:
        #     raise ValueError(f"Explicit scheme `{self.scheme}` is not supported")

        if self.info["backend"] == "numba":
            fixed_stepper = jit(fixed_stepper)  # compile inner stepper

        def wrapped_stepper(state: FieldBase, t_start: float, t_end: float) -> float:
            """advance `state` from `t_start` to `t_end` using fixed steps"""
            # calculate number of steps (which is at least 1)
            steps = max(1, int(np.ceil((t_end - t_start) / dt)))

            # distribute the number of steps and the field to all nodes
            steps = self.mesh.broadcast(steps)
            sub_state_data = self.mesh.split_field_data_mpi(state.data)

            # evolve the sub state
            t_last, modifications = fixed_stepper(sub_state_data, t_start, steps)

            # collect the data from all nodes
            modification_list = self.mesh.gather(modifications)
            self.mesh.combine_field_data_mpi(sub_state_data, out=state.data)

            # store information in the main node
            if mpi.is_main:
                self.info["steps"] += steps
                self.info["state_modifications"] += sum(modification_list)
            return t_last

        return wrapped_stepper
