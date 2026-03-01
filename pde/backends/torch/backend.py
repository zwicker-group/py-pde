"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numbers
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...grids import GridBase
from ..base import OperatorInfo, TFunc
from ..numpy import NumpyBackend
from .utils import NUMPY_TO_TORCH_DTYPE, TORCH_TO_NUMPY_DTYPE

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ...grids import GridBase
    from ...grids.boundaries.axes import BoundariesBase
    from ...pdes import PDEBase
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import (
        NumberOrArray,
        NumericArray,
        TArray,
        TField,
    )
    from ..base import TFunc
    from ..numpy.backend import OperatorInfo
    from ..registry import BackendRegistry
    from .utils import TorchDifferentialOperatorType


class TorchBackend(NumpyBackend):
    """Defines :mod:`torch` backend."""

    compile_options = {
        "fullgraph": True,
        "dynamic": False,
        "options": {"epilogue_fusion": True, "max_autotune": True},
    }
    """dict: defines options that affect compilation by torch"""

    _dtype_cache: dict[str, dict[DTypeLike, torch.dtype]] = defaultdict(dict)

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

    def get_torch_dtype(self, dtype: DTypeLike) -> torch.dtype:
        """Convert dtype to torch dtype.

        Args:
            dtype:
                numpy dtype to convert to corresponding torch dtype

        Returns:
            :class:`torch.dtype`:
                A proper dtype for torch
        """
        if isinstance(dtype, torch.dtype):
            return dtype

        # load the dtype cache of the current device
        type_cache = self._dtype_cache[self.device.type]
        np_dtype = np.dtype(dtype)
        try:
            # try returning type from cache
            return type_cache[np_dtype]
        except KeyError:
            pass

        # convert numpy dtype to corresponding torch dtype
        torch_dtype = NUMPY_TO_TORCH_DTYPE[dtype]

        try:
            # Try to create a tensor of this dtype on the device
            torch.empty(1, dtype=torch_dtype, device=self.device)
        except TypeError:
            # dtype is not supported
            if self.config["dtype_downcasting"] and torch_dtype == torch.float64:
                # TODO: Add warning
                torch_dtype = torch.float32
            else:
                raise

        # store dtype in cache
        type_cache[np_dtype] = torch_dtype
        return torch_dtype

    def get_numpy_dtype(self, dtype: DTypeLike) -> np.dtype:
        """Determine numpy dtype suitable for the torch backend.

        Args:
            dtype:
                numpy dtype to convert to supported dtype

        Returns:
            :class:`torch.dtype`:
                A numpy dtype that is compatible with the torch backend
        """
        return TORCH_TO_NUMPY_DTYPE[self.get_torch_dtype(dtype)]

    def from_numpy(self, value: Any) -> Any:
        """Convert values from numpy to torch representation.

        This method also ensures that the value is copied to the selected device.
        """
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, np.ndarray):
            arr_torch = torch.from_numpy(value)
            return arr_torch.to(self.device, dtype=self.get_torch_dtype(value.dtype))
        if isinstance(value, numbers.Number):
            return value
        msg = f"Unsupported type `{value.__type__}"
        raise TypeError(msg)

    def to_numpy(self, value: Any) -> Any:
        """Convert native values to numpy representation."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        return value

    def compile_function(self, func: TFunc, **compile_options) -> TFunc:
        r"""General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
            **compile_options:
                Additional keyword arguments will be forwarded to :func:`torch.compile`
        """
        if not self.config["compile"]:
            return func

        # compile the function using the torch backend
        opts = self.compile_options | compile_options
        return torch.compile(func, **opts)  # type: ignore

    def make_operator_no_bc(  # type: ignore
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        native: bool = False,
        **kwargs,
    ) -> TorchDifferentialOperatorType:
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
            native (bool):
                If True, the returned functions expects the native data representation
                of the backend. Otherwise, the input and output are expected to be
                :class:`~numpy.ndarray`.
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray), so they `out` array need
            to be supplied explicitly.
        """
        # obtain details about the operator
        operator_info = self.get_operator_info(grid, operator)
        dtype = self.get_numpy_dtype(kwargs.pop("dtype", np.double))

        # create an operator with or without BCs
        torch_operator = operator_info.factory(grid, bcs=None, dtype=dtype, **kwargs)
        torch_operator.eval()  # type: ignore

        # compile the function and move it to the device
        torch_operator_jitted = self.compile_function(torch_operator)
        torch_operator_jitted.to(self.device)  # type: ignore

        if native:
            return torch_operator_jitted  # type: ignore

        def operator_no_bc(arr: NumericArray, out: NumericArray) -> None:
            arr_torch = self.from_numpy(arr)
            out_torch = torch_operator_jitted(arr_torch)  # type: ignore
            out[...] = self.to_numpy(out_torch)

        return operator_no_bc  # type: ignore

    def make_operator(  # type: ignore
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        native: bool = False,
        **kwargs,
    ) -> TorchDifferentialOperatorType:
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
            native (bool):
                If True, the returned functions expects the native data representation
                of the backend. Otherwise, the input and output are expected to be
                :class:`~numpy.ndarray`.
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Warning:
            The same operator should not be assigned to different variables that are
            used in the same code, because :mod:`torch` has problems compiling the
            resulting code. This particularly precludes caching the operators, since
            they then might be reused, e.g., if boundary conditions agree between
            different operators.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # obtain details about the operator
        operator_info = self.get_operator_info(grid, operator)
        dtype = self.get_numpy_dtype(kwargs.pop("dtype", np.double))
        bcs = grid.get_boundary_conditions(bcs, rank=operator_info.rank_in)

        # create an operator with or without BCs
        torch_operator = operator_info.factory(grid, bcs, dtype=dtype, **kwargs)  # type: ignore
        torch_operator.eval()  # type: ignore

        # compile the function and move it to the device
        torch_operator_jitted = self.compile_function(torch_operator)
        torch_operator_jitted.to(self.device)  # type: ignore

        if native:
            # return the native representation if requested
            return torch_operator_jitted  # type: ignore

        # wrap the operator such that it can be called from numpy
        shape_out = (grid.dim,) * operator_info.rank_out + grid.shape

        # define numpy version of the operator
        def apply_op(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            arr_torch = self.from_numpy(arr)

            if out is None:
                out = np.empty(shape_out, dtype=arr.dtype)
            elif out.shape != shape_out:
                msg = f"Incompatible shapes {out.shape} != {shape_out}"
                raise ValueError(msg)
            out_torch = torch_operator_jitted(arr_torch)  # type: ignore
            out[:] = self.to_numpy(out_torch)
            return out

        # return the compiled versions of the operator
        return apply_op  # type: ignore

    def make_integrator(
        self, grid: GridBase, *, dtype: DTypeLike = np.double
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined
            dtype:
                The data type of the field that is being integrated

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        from .operators.common import TorchIntegralOperator

        # create the torch operator
        integrate_torch = self.compile_function(
            TorchIntegralOperator(grid, dtype=self.get_numpy_dtype(dtype))
        )
        integrate_torch.to(self.device)

        def integrate_global(arr: NumericArray) -> NumberOrArray:
            """Integrate data.

            Args:
                arr (:class:`~numpy.ndarray`): discretized data on grid
            """
            # move data to device
            arr_torch = self.from_numpy(arr)
            # integrate on device
            res = integrate_torch(arr_torch)
            # return result
            res_np = self.to_numpy(res)
            if res_np.ndim == 0:
                return res_np[()]  # type: ignore
            return res_np  # type: ignore

        return integrate_global

    def make_pde_rhs(
        self, eq: PDEBase, state: TField, *, native: bool = False
    ) -> Callable[[TArray, float], TArray]:
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted
            native (bool):
                If True, the returned functions expects the native data representation
                of the backend. Otherwise, the input and output are expected to be
                :class:`~numpy.ndarray`.

        Returns:
            Function returning deterministic part of the right hand side of the PDE.
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
        if native:
            return rhs_torch  # type: ignore

        def rhs(arr: NumericArray, t: float = 0) -> NumericArray:
            """Helper wrapping function working with torch tensors."""
            arr_torch = self.from_numpy(arr)
            res_torch = rhs_torch(arr_torch, t)
            return self.to_numpy(res_torch)  # type: ignore

        return rhs  # type: ignore

    def make_expression_function(
        self,
        expression: ExpressionBase,
        *,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
    ) -> Callable[..., NumberOrArray]:
        """Return a function evaluating an expression.

        Args:
            expression (:class:`~pde.tools.expression.ExpressionBase`):
                The expression that is converted to a function
            single_arg (bool):
                Determines whether the returned function accepts all variables in a
                single argument as an array or whether all variables need to be
                supplied separately.
            user_funcs (dict):
                Additional functions that can be used in the expression.

        Returns:
            function: the function
        """
        import sympy
        from sympy.printing.pycode import PythonCodePrinter

        from ...tools.expressions import SPECIAL_FUNCTIONS

        # collect all the user functions
        user_functions = expression.user_funcs.copy()
        if user_funcs is not None:
            user_functions.update(user_funcs)
        user_functions.update(SPECIAL_FUNCTIONS)

        user_functions = {
            k: self.compile_function(v) for k, v in user_functions.items()
        }

        # initialize the printer that deals with numpy arrays correctly

        class ListArrayPrinter(PythonCodePrinter):
            """Special sympy printer returning arrays as lists."""

            def _print_ImmutableDenseNDimArray(self, arr):
                arrays = ", ".join(f"{self._print(expr)}" for expr in arr)
                return f"[{arrays}]"

        printer = ListArrayPrinter(
            {
                "fully_qualified_modules": False,
                "inline": True,
                "allow_unknown_functions": True,
                "user_functions": {k: k for k in user_functions},
            }
        )

        # determine the list of variables that the function depends on
        variables = (expression.vars,) if single_arg else tuple(expression.vars)
        constants = tuple(expression.consts)

        # turn the expression into a callable function
        self._logger.info("Parse sympy expression `%s`", expression._sympy_expr)
        func = sympy.lambdify(
            variables + constants,
            expression._sympy_expr,
            modules=[user_functions, "torch"],
            printer=printer,
        )

        # Apply the constants if there are any. Note that we use this pattern of a
        # partial function instead of replacing the constants in the sympy expression
        # directly since sympy does not work well with numpy arrays.
        if constants:
            const_values = tuple(
                self.from_numpy(expression.consts[c]) for c in constants
            )

            func = self.compile_function(func)

            def result(*args):
                return func(*args, *const_values)

        else:
            result = func
        return self.compile_function(result)
