"""
Defines an explicit solver using multiprocessing via MPI
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, List, Union

import numba as nb
import numpy as np

from ..fields.base import FieldBase
from ..grids.mesh import GridMesh
from ..pdes.base import PDEBase
from ..tools import mpi
from ..tools.numba import jit
from .explicit import ExplicitSolver


class ExplicitMPISolver(ExplicitSolver):
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
        scheme: str = "euler",
        decomposition: Union[int, List[int]] = -1,
        *,
        backend: str = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The instance describing the pde that needs to be solved
            scheme (str):
                Defines the explicit scheme to use. Supported values are 'euler' and
                'runge-kutta' (or 'rk' for short).
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
        super().__init__(pde, scheme=scheme, backend=backend, adaptive=False)
        self.decomposition = decomposition

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
        sub_state = self.mesh.extract_subfield(state)
        self.info["grid_decomposition"] = self.mesh.shape

        if self.adaptive:
            self._logger.warning(
                "Adaptive stochastic stepping is not implemented. Using a fixed time "
                "step instead."
            )

        # create stepper with fixed steps
        if self.scheme == "euler":
            fixed_stepper = self._make_fixed_euler_stepper(sub_state, dt)
        elif self.scheme in {"runge-kutta", "rk", "rk45"}:
            fixed_stepper = self._make_rk45_stepper(sub_state, dt)
        else:
            raise ValueError(f"Explicit scheme `{self.scheme}` is not supported")

        if self.backend == "numba":
            sig = (nb.typeof(sub_state.data), nb.double, nb.int_)
            fixed_stepper = jit(sig)(fixed_stepper)  # compile inner stepper

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
                self.info["state_modifications"] += sum(modification_list)  # type: ignore
            return t_last

        return wrapped_stepper
