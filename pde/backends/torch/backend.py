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
    from ...grids.boundaries.axes import BoundariesData
    from ...pdes import PDEBase
    from ...tools.typing import (
        NumberOrArray,
        NumericArray,
        OperatorImplType,
        OperatorType,
        TField,
    )
    from ..base import TFunc
    from ..numpy.backend import OperatorInfo
    from ..registry import BackendRegistry
    from .utils import TorchOperatorType


class TorchBackend(NumpyBackend):
    """Defines :mod:`torch` backend."""

    compile_options = {
        "fullgraph": True,
        "dynamic": False,
        "options": {"epilogue_fusion": True, "max_autotune": True},
    }

    def __init__(self, name: str, registry: BackendRegistry, *, device: str = "config"):
        """Initialize the torch backend.

        Args:
            registry (:class:`~pde.backends.registry.BackendRegistry`):
                The registry to which this backend is added
            name (str):
                The name of the backend
            device (str):
                The torch device to use. Special values are "config" (read from
                configuration) and "auto" (use CUDA if available, otherwise CPU)
        """
        super().__init__(name=name, registry=registry)

        self.device = device

    @property
    def device(self) -> torch.device:
        """The currently assigned torch device."""
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set a new torch device."""
        if device == "config":
            device = self.config["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

    def compile_function(self, func: TFunc, **compile_options) -> TFunc:
        r"""General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
            **compile_options:
                Additional keyword arguments will be forwarded to :func:`torch.compile`
        """
        opts = self.compile_options | compile_options
        return torch.compile(func, **opts)  # type: ignore

    def make_torch_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesData | None,
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
        parameters, like time.

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
        torch_operator = self.make_torch_operator(grid, operator, bcs=None, **kwargs)

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
        parameters, like time.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        torch_operator = self.make_torch_operator(grid, operator, bcs, **kwargs)
        torch_operator_jitted = self.compile_function(torch_operator)
        torch_operator_jitted.to(self.device)  # type: ignore

        shape_out = (grid.dim,) * operator_info.rank_out + grid.shape

        # define numpy version of the operator
        def apply_op(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            arr_torch = torch.from_numpy(arr)
            arr_torch.to(self.device)

            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            elif out.shape != shape_out:
                msg = f"Incompatible shapes {out.shape} != {shape_out}"
                raise ValueError(msg)
            out[:] = torch_operator_jitted(arr_torch)
            return out

        # return the compiled versions of the operator
        return apply_op

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
        from .operators.common import IntegralOperator

        # create the torch operator
        integrate_torch = self.compile_function(IntegralOperator(grid))
        integrate_torch.to(self.device)

        def integrate_global(arr: NumericArray) -> float:
            """Integrate data.

            Args:
                arr (:class:`~numpy.ndarray`): discretized data on grid
            """
            # move data to device
            arr_torch = torch.from_numpy(arr)
            arr_torch.to(self.device)
            # integrate on device
            res = integrate_torch(arr_torch)
            # return result
            res_np = res.cpu().detach().numpy()
            if res_np.ndim == 0:
                return res_np[()]  # type: ignore
            return res_np  # type: ignore

        return integrate_global

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

        # get the compiled right hand side
        rhs_torch = self.compile_function(make_rhs(state))

        def rhs(arr: NumericArray, t: float = 0) -> NumericArray:
            arr_torch = torch.from_numpy(arr)
            arr_torch.to(self.device)

            return rhs_torch(arr_torch).numpy()  # type: ignore

        return rhs
