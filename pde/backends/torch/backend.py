"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numbers
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from ...fields import VectorField
from ...grids import GridBase
from ..base import BackendBase, OperatorInfo, TFunc
from .utils import NUMPY_TO_TORCH_DTYPE, TORCH_TO_NUMPY_DTYPE

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ...fields import DataFieldBase
    from ...grids import GridBase
    from ...grids.boundaries.axes import BoundariesBase
    from ...pdes import PDEBase
    from ...solvers.base import SolverBase
    from ...tools.config import Config
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import NumberOrArray, NumericArray, TField, TNativeArray
    from ..base import TFunc
    from ..numpy.backend import OperatorInfo
    from .operators.common import TorchDifferentialOperator


class TorchBackend(BackendBase):
    """Defines :mod:`torch` backend."""

    implementation = "torch"
    copy_data = True

    compile_options = {
        "fullgraph": True,  # force compilation of entire graph (no graph breaks)
        "dynamic": False,  # compiled functions do not support changing shapes
        "backend": "inductor",  # use compiled optimized kernels for speed
        "options": {"epilogue_fusion": True, "max_autotune": True},
    }
    """dict: defines options that affect compilation by torch"""
    _dtype_cache: dict[str, dict[DTypeLike, torch.dtype]] = defaultdict(dict)
    """dict: contains information about the dtypes available for the current device"""
    _emitted_downcast_warning: bool = False
    """bool: global flag to track whether we already warned about downcasting"""

    def __init__(
        self,
        config: Config | None = None,
        *,
        name: str | None = None,
        device: str = "config",
    ):
        """Initialize the torch backend.

        Args:
            config (:class:`~pde.tools.config.Config`):
                Configuration data for the backend
            name (str):
                The name of the backend
            device (str):
                The torch device to use. Special values are "config" (read from
                configuration) and "auto" (use CUDA if available, otherwise CPU)
        """
        if config is None:
            from .config import DEFAULT_CONFIG as config  # type: ignore

        super().__init__(config, name=name)
        self.device = device

    def __repr__(self) -> str:
        """Return concise string representation of this backend."""
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"device={str(self.device)!r})"
        )

    @property
    def device(self) -> torch.device:
        """The currently assigned torch device."""
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set a new torch device."""
        # determine which device we need to use
        if device == "config":
            device = self.config["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # check whether the device is available
        if device.startswith("cuda") and not torch.cuda.is_available():
            msg = "cuda device is not available"
            raise RuntimeError(msg)
        if device.startswith("mps") and not torch.backends.mps.is_available():
            msg = "mps device is not available"
            raise RuntimeError(msg)

        # set the actual device
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
        torch_dtype = NUMPY_TO_TORCH_DTYPE[np_dtype]

        try:
            # Try to create a tensor of this dtype on the device
            torch.empty(1, dtype=torch_dtype, device=self.device)
        except TypeError:
            # dtype is not supported, so we see whether we need to use downcasting
            if self.config["dtype_downcasting"] and torch_dtype == torch.float64:
                if not self._emitted_downcast_warning:
                    self._logger.warning(
                        " %s device doesn't support float64, so we use float32 instead",
                        self.device.type,
                    )
                    self._emitted_downcast_warning = True
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
        # convert numpy dtype to torch dtype to support cases where the torch device
        # only supports narrower types
        return TORCH_TO_NUMPY_DTYPE[self.get_torch_dtype(dtype)]

    def from_numpy(self, value: Any) -> Any:
        """Convert values from numpy to torch representation.

        This method also ensures that the value is copied to the selected device.
        """
        if isinstance(value, torch.Tensor):
            return value.to(self.device)  # move tensor to device

        if isinstance(value, (np.ndarray, numbers.Number)):
            value_arr = np.asarray(value)  # convert numbers to arrays for torch
            arr_torch = torch.from_numpy(value_arr)  # convert to torch.Tensor
            dtype = self.get_torch_dtype(value_arr.dtype)
            return arr_torch.to(self.device, dtype=dtype)  # move tensor to device

        msg = f"Unsupported type `{type(value).__name__}"
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

    def make_operator_no_bc(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        dtype: DTypeLike | None = None,
        native: bool = False,
        **kwargs,
    ) -> TorchDifferentialOperator:
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
            dtype (numpy dtype):
                The data type of the field.
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
        dtype = self.get_numpy_dtype(dtype or np.double)

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

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        dtype: DTypeLike | None = None,
        native: bool = False,
        **kwargs,
    ) -> TorchDifferentialOperator:
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
            dtype (numpy dtype):
                The data type of the field.
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
        dtype = self.get_numpy_dtype(dtype or np.double)
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

    def make_integrator(  # type: ignore
        self, grid: GridBase, *, dtype: DTypeLike = np.double
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return function that integrates discretized data over a grid.

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

        return integrate_torch

    def make_inner_prod_operator(  # type: ignore
        self, field: DataFieldBase, *, conjugate: bool = True
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the inner product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        num_axes = field.grid.num_axes

        def dot(
            a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Numpy implementation to calculate dot product between two fields."""
            rank_a = a.ndim - num_axes
            rank_b = b.ndim - num_axes
            if rank_a < 1 or rank_b < 1:
                msg = "Fields in dot product must have rank >= 1"
                raise TypeError(msg)
            if a.shape[rank_a:] != b.shape[rank_b:]:
                msg = "Shapes of fields are not compatible for dot product"
                raise ValueError(msg)
            if out is not None:
                msg = "torch implementation of inner product does not allow `out` arg."
                raise TypeError(msg)

            if conjugate:
                b = b.conj()

            if rank_a == 1 and rank_b == 1:  # result is scalar field
                return torch.einsum("i...,i...->...", a, b)

            if rank_a == 2 and rank_b == 1:  # result is vector field
                return torch.einsum("ij...,j...->i...", a, b)

            if rank_a == 1 and rank_b == 2:  # result is vector field
                return torch.einsum("i...,ij...->j...", a, b)

            if rank_a == 2 and rank_b == 2:  # result is tensor-2 field
                return torch.einsum("ij...,jk...->ik...", a, b)

            msg = f"Unsupported shapes ({a.shape}, {b.shape})"
            raise TypeError(msg)

        return dot

    def make_outer_prod_operator(  # type: ignore
        self, field: DataFieldBase
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
        """Return operator calculating the outer product between two fields.

        This supports typically only supports products between two vector fields.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the outer product is defined

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        if not isinstance(field, VectorField):
            msg = "Can only define outer product between vector fields"
            raise TypeError(msg)

        def outer(
            a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Calculate the outer product using numpy."""
            if out is not None:
                msg = "torch implementation of inner product does not allow `out` arg."
                raise TypeError(msg)
            return torch.einsum("i...,j...->ij...", a, b)

        return outer

    def make_pde_rhs(
        self, eq: PDEBase, state: TField, *, native: bool = False
    ) -> Callable[[TNativeArray, float], TNativeArray]:
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
        # the following method is deprecated since 2026-03-02
        try:
            make_rhs = eq.make_pde_rhs_torch  # type: ignore
        except AttributeError:
            # method is not implemented, which should be the default
            rhs_native = None
        else:
            warnings.warn(
                "`eq.make_pde_rhs_torch` method is deprecated. Implement "
                "`eq.make_evolution_rate` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            rhs_native = make_rhs(state)

        if rhs_native is None:
            try:
                make_rhs = eq.make_evolution_rate
            except AttributeError as err:
                msg = (
                    "The right-hand side of the PDE is not implemented using the "
                    f"`{self.name}` backend. To add the implementation, provide the "
                    "method `make_evolution_rate`, which should return a compilable "
                    "function calculating the evolution rate."
                )
                raise NotImplementedError(msg) from err
            else:
                rhs_native = make_rhs(state, backend=self)

        # get the compiled right hand side
        rhs_torch = self.compile_function(rhs_native)
        if native:
            return rhs_torch

        def rhs(arr: NumericArray, t: float = 0) -> NumericArray:
            """Helper wrapping function working with torch tensors."""
            arr_torch = self.from_numpy(arr)
            # We wrap the scalar time into a tensor, so torch correctly identifies it as
            # a value that is modified each time we call the function.
            t_torch = self.from_numpy(t)
            res_torch = rhs_torch(arr_torch, t_torch)
            return self.to_numpy(res_torch)  # type: ignore

        return rhs  # type: ignore

    def make_inner_stepper(
        self,
        solver: SolverBase,
        stepper_style: Literal["fixed", "adaptive"],
        state: TField,
        dt: float,
    ) -> Callable:
        """Return a stepper function using an explicit scheme.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            stepper_style (str):
                The style of the stepper, either "fixed" or "adaptive"
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step used (Uses :attr:`SolverBase.dt_default` if `None`)

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        solver.info["backend"]["device"] = self.device.type
        solver.info["backend"]["compile"] = self.config["compile"]
        return super().make_inner_stepper(solver, stepper_style, state, dt)

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

        from .utils import SPECIAL_FUNCTIONS_TORCH

        # collect all the user functions
        user_functions = expression.user_funcs.copy()
        if user_funcs is not None:
            user_functions.update(user_funcs)
        user_functions.update(SPECIAL_FUNCTIONS_TORCH)

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
