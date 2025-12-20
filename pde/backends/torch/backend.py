"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...grids import GridBase
from ...grids.boundaries.axes import BoundariesBase
from ..base import OperatorInfo, TFunc
from ..numpy import NumpyBackend
from .utils import AnyDType, get_torch_dtype

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...grids import BoundariesBase, GridBase
    from ...pdes import PDEBase
    from ...tools.typing import NumericArray, OperatorImplType, OperatorType, TField
    from ..base import TFunc
    from ..numpy.backend import OperatorInfo
    from .utils import TorchOperatorType


class TorchBackend(NumpyBackend):
    """Defines :mod:`torch` backend."""

    compile_options = {
        "fullgraph": True,
        "dynamic": False,
        "options": {"epilogue_fusion": True, "max_autotune": True},
    }

    def __init__(self, name: str = "", device: str = "auto"):
        super().__init__(name=name)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        return torch.compile(func, **self.compile_options)  # type: ignore

    def make_torch_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesBase | None,
        dtype: AnyDType = np.double,
        **kwargs,
    ) -> TorchOperatorType:
        """Return a torch function applying an operator with boundary conditions.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                The boundary conditions used before the operator is applied
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array.

        The function also accepts an optional parameter `args`, which is forwarded to
        `set_ghost_cells`. This allows setting boundary conditions based on external
        parameters, like time. Note that since the returned operator will always be
        compiled by Numba, the arguments need to be compatible with Numba. The
        following example shows how to pass the current time `t`:

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        if bcs is not None:
            bcs = grid.get_boundary_conditions(bcs, rank=operator_info.rank_in)
        return operator_info.factory(grid, bcs, dtype=get_torch_dtype(dtype), **kwargs)  # type: ignore

    def make_operator_no_bc(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        **kwargs,
    ) -> OperatorImplType:
        """Return a compiled function applying an operator without boundary conditions.

        A function that takes the discretized full data as an input and an array of
        valid data points to which the result of applying the operator is written.

        Note:
            The resulting function does not check whether the ghost cells of the input
            array have been supplied with sensible values. It is the responsibility of
            the user to set the values of the ghost cells beforehand. Use this function
            only if you absolutely know what you're doing. In all other cases,
            :meth:`make_operator` is probably the better choice.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray), so they `out` array need
            to be supplied explicitly.
        """
        # determine the operator for the chosen backend
        torch_operator = self.make_torch_operator(grid, operator, bcs=None)

        def operator_no_bc(arr: NumericArray, out: NumericArray) -> None:
            arr_torch = torch.from_numpy(arr)
            out[...] = torch_operator(arr_torch)

        return operator_no_bc

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesBase,
        **kwargs,
    ) -> OperatorType:
        """Return a compiled function applying an operator with boundary conditions.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                The boundary conditions used before the operator is applied
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array.

        The function also accepts an optional parameter `args`, which is forwarded to
        `set_ghost_cells`. This allows setting boundary conditions based on external
        parameters, like time. Note that since the returned operator will always be
        compiled by Numba, the arguments need to be compatible with Numba. The
        following example shows how to pass the current time `t`:

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        torch_operator = self.make_torch_operator(grid, operator, bcs)
        torch_operator_jitted = torch.compile(torch_operator, **self.compile_options)  # type: ignore
        torch_operator_jitted.to(self.device)

        shape_out = (grid.dim,) * operator_info.rank_out + grid.shape

        # define numpy version of the operator
        def apply_op(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            torch_arr = torch.from_numpy(arr)
            torch_arr.to(self.device)

            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            elif out.shape != shape_out:
                msg = f"Incompatible shapes {out.shape} != {shape_out}"
                raise ValueError(msg)
            out[:] = torch_operator_jitted(torch_arr)
            return out

        # return the compiled versions of the operator
        return apply_op

    def make_pde_rhs(
        self, eq: PDEBase, state: TField
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE
        """
        try:
            make_rhs = eq.make_pde_rhs_torch  # type: ignore
        except AttributeError as err:
            msg = (
                "The right-hand side of the PDE is not implemented using the `torch` "
                "backend. To add the implementation, provide the method "
                "`make_pde_rhs_torch`, which should return a compilable function "
                "calculating the evolution rate using a torch array as input."
            )
            raise NotImplementedError(msg) from err
        return self.compile_function(make_rhs(state))  # type: ignore

    # def make_inner_stepper(
    #     self,
    #     solver: SolverBase,
    #     stepper_style: Literal["fixed", "adaptive"],
    #     state: TField,
    #     dt: float,
    # ) -> Callable:
    #     """Return a stepper function using an explicit scheme.

    #     Args:
    #         solver (:class:`~pde.solvers.base.SolverBase`):
    #             The solver instance, which determines how the stepper is constructed
    #         state (:class:`~pde.fields.base.FieldBase`):
    #             An example for the state from which the grid and other information can
    #             be extracted
    #         dt (float):
    #             Time step used (Uses :attr:`SolverBase.dt_default` if `None`)

    #     Returns:
    #         Function that can be called to advance the `state` from time `t_start` to
    #         time `t_end`. The function call signature is `(state: numpy.ndarray,
    #         t_start: float, t_end: float)`
    #     """
    #     assert solver.backend == self.name

    #     from ._solvers import make_adaptive_stepper, make_fixed_stepper

    #     solver.info["dt_statistics"] = OnlineStatistics()

    #     if stepper_style == "fixed":
    #         return make_fixed_stepper(solver, state, dt=dt)
    #     if stepper_style == "adaptive":
    #         assert isinstance(solver, AdaptiveSolverBase)
    #         return make_adaptive_stepper(solver, state)
    #     raise NotImplementedError
