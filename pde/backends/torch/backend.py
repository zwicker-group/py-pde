"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numbers
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...fields import VectorField
from ...grids import GridBase
from ...solvers.scipy import ScipySolver
from ..base import BackendBase, OperatorInfo
from .typing import NUMPY_TO_TORCH_DTYPE, TORCH_TO_NUMPY_DTYPE, TorchRHSType

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ...fields import DataFieldBase
    from ...grids import GridBase
    from ...grids.boundaries.axes import BoundariesBase
    from ...pdes import PDEBase
    from ...solvers import SolverBase
    from ...tools.config import ConfigLike
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import NumberOrArray, NumericArray, StepperType, TField, TFunc
    from ..numpy.backend import OperatorInfo
    from .operators.common import TorchDifferentialOperator


class TorchBackend(BackendBase[torch.Tensor]):
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
        config: ConfigLike | None = None,
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
            from .config import DEFAULT_CONFIG as config

        super().__init__(config, name=name)
        self.device = device

    @classmethod
    def from_args(
        cls, config: ConfigLike | None, args: str = "", *, name: str | None = None
    ):
        """Initialize backend with extra arguments.

        Args:
            config (:class:`~pde.tools.config.Config`):
                Configuration data for the backend
            args (str):
                Additional arguments that determine how the backend is initialized
            name (str):
                The name of the backend
        """
        return cls(config, name=name, device=args)

    def __repr__(self) -> str:
        """Return concise string representation of this backend."""
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"device={str(self.device)!r})"
        )

    @property
    def info(self) -> dict[str, Any]:
        """dict: relevant information about the backend"""
        info = super().info
        info["device"] = self.device.type
        info["compile"] = self._config_parameter("compile")
        return info

    @property
    def device(self) -> torch.device:
        """The currently assigned torch device."""
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set a new torch device."""
        # determine which device we need to use
        if device == "config":
            device = self._config_parameter("device")
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
            if (
                self._config_parameter("dtype_downcasting")
                and torch_dtype == torch.float64
            ):
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

    def numpy_to_native(self, value: Any) -> torch.Tensor:  # type: ignore
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

    def native_to_numpy(self, value: Any) -> Any:
        """Convert native values to numpy representation."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        return value

    def compile_function(
        self, func: TFunc, *, to_device: bool = False, **compile_options
    ) -> TFunc:
        r"""General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
            to_device (bool):
                Moves (compiled) function to device
            **compile_options:
                Additional keyword arguments will be forwarded to :func:`torch.compile`
        """
        if self._config_parameter("compile"):
            # compile the function using the torch backend
            opts = self.compile_options | compile_options
            func = torch.compile(func, **opts)  # type: ignore
        if to_device and isinstance(func, torch.nn.Module):
            func.to(self.device)  # move module to correct device
        return func

    def _apply_operator(
        self, func: Callable, *values: NumericArray, out: NumericArray, **kwargs
    ) -> None:
        r"""Apply a native operator to numpy data.

        Args:
            func (callable):
                The operator defined in the native space of the backend
            values (:class:`~numpy.ndarray`):
                The array data that is fed to the function
            out (:class:`~numpy.ndarray`):
                The array to which the results are written
            *args, **kwargs:
                Additional arguments that are forwarded to the function call
        """
        values_native = [self.numpy_to_native(value) for value in values]
        out_native = func(*values_native, **kwargs)
        out[...] = self.native_to_numpy(out_native)

    def make_operator_no_bc(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        dtype: DTypeLike | None = None,
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
        torch_operator: torch.nn.Module = operator_info.factory(  # type: ignore
            grid, bcs=None, dtype=dtype, **kwargs
        )
        torch_operator.eval()

        # compile the function and move it to the device
        return self.compile_function(torch_operator, to_device=True)  # type: ignore

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        dtype: DTypeLike | None = None,
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
        torch_operator: torch.nn.Module = operator_info.factory(  # type: ignore
            grid, bcs, dtype=dtype, **kwargs
        )
        torch_operator.eval()

        # compile the function and move it to the device
        return self.compile_function(torch_operator, to_device=True)  # type: ignore

    def make_integrator(
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
        return self.compile_function(
            TorchIntegralOperator(grid, dtype=self.get_numpy_dtype(dtype)),
            to_device=True,
        )

    def make_inner_prod_operator(
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

    def make_outer_prod_operator(
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

    def make_pde_rhs(self, eq: PDEBase, state: TField) -> TorchRHSType:  # type: ignore
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE.
        """
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
        return self.compile_function(rhs_native)  # type: ignore

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
                self.numpy_to_native(expression.consts[c]) for c in constants
            )

            func = self.compile_function(func)

            def result(*args):
                return func(*args, *const_values)

        else:
            result = func
        return self.compile_function(result, to_device=True)

    def make_gaussian_noise(
        self, field: TField, *, rng: np.random.Generator
    ) -> Callable[[], torch.Tensor]:
        """Create a function generating Gaussian white noise.

        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used to initialize the seed.
        """
        from .utils import TorchGaussianNoise

        data_shape: tuple[int, ...] = field.data.shape
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(rng.integers(0, 2**32)))

        return TorchGaussianNoise(
            data_shape, dtype=self.get_numpy_dtype(field.dtype), generator=generator
        ).to(self.device)

    def make_stepper(self, solver: SolverBase, state: TField) -> StepperType:
        """Create a field-based stepping function for a given solver.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        from ._solvers import make_inner_stepper

        assert solver.backend == self
        if isinstance(solver, ScipySolver):
            msg = "Torch backend does not support Scipy solver"
            raise NotImplementedError(msg)

        # create the backend-level stepping function
        inner_stepper = make_inner_stepper(solver, state)

        def stepper(state: TField, t_start: float, t_end: float) -> float:
            """Advance `state` by executing the backend-level stepping function."""
            # push state data to native backend
            state_tensor: torch.Tensor = solver.backend.numpy_to_native(state.data)
            # execute the backend-level stepping function
            state_tensor, t_last = inner_stepper(state_tensor, t_start, t_end)
            # retrieve data from native backend
            state.data[:] = solver.backend.native_to_numpy(state_tensor)
            return t_last

        return stepper  # type: ignore
