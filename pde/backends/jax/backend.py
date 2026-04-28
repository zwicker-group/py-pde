"""Defines the :mod:`jax` backend class.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numbers
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ...fields import VectorField
from ...grids import GridBase
from ...grids.boundaries.axes import BoundariesList
from ...tools.cache import cached_method
from ..base import BackendBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ...fields import DataFieldBase
    from ...grids import GridBase
    from ...grids.boundaries.axes import BoundariesBase
    from ...grids.boundaries.local import BCBase
    from ...pdes import PDEBase
    from ...solvers import SolverBase
    from ...tools.config import Config, ConfigLike
    from ...tools.expressions import ExpressionBase
    from ...tools.typing import (
        NumberOrArray,
        NumericArray,
        OperatorImplType,
        OperatorInfo,
        OperatorType,
        StepperType,
        TField,
        TFunc,
    )
    from .typing import JaxDataSetter, JaxGhostCellSetter


class JaxBackend(BackendBase[jax.Array]):
    """Defines :mod:`jax` backend."""

    implementation = "jax"
    copy_data = True

    _dtype_cache: dict[str, dict[DTypeLike, DTypeLike]] = defaultdict(dict)
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
        """Initialize the jax backend.

        Args:
            config (:class:`~pde.tools.config.Config`):
                Configuration data for the backend
            name (str):
                The name of the backend
            device (str):
                The jax device to use. Special values are "config" (read from
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

    @property
    def info(self) -> dict[str, Any]:
        """dict: relevant information about the backend"""
        info = super().info
        info["device"] = self.device.device_kind
        info["compile"] = self.config["compile"]
        return info

    @property
    def device(self) -> jax.Device:
        """The currently assigned jax device."""
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set a new jax device."""
        # determine which device we need to use
        if device == "config":
            device = self.config["device"]
        if device == "auto":
            try:
                self._device = jax.devices("gpu")[0]
            except RuntimeError:
                self._device = jax.devices("cpu")[0]
        elif ":" in device:
            name, dev_id = device.split(":", 1)
            self._device = jax.devices(name)[int(dev_id)]
        else:
            self._device = jax.devices(device)[0]

    def get_jax_dtype(self, dtype: DTypeLike) -> DTypeLike:
        """Convert numpy dtype to jax-compatible dtype.

        Args:
            dtype:
                numpy dtype to convert to corresponding jax dtype

        Returns:
            :class:`np.dtype`:
                A proper dtype usable for jax
        """
        # load the dtype cache of the current device
        type_cache = self._dtype_cache[self.device.device_kind]
        np_dtype: DTypeLike = np.dtype(dtype)
        try:
            # try returning type from cache
            return type_cache[np_dtype]
        except KeyError:
            pass

        # determine jax_dtype
        jax_dtype = jax.dtypes.canonicalize_dtype(np_dtype)

        # store dtype in cache
        type_cache[np_dtype] = jax_dtype
        return jax_dtype

    def numpy_to_native(self, value: Any) -> Any:
        """Convert values from numpy to jax representation.

        This method also ensures that the value is copied to the selected device.
        """
        if isinstance(value, (jax.Array, numbers.Number)):
            return jax.device_put(value, self.device)
        if isinstance(value, np.ndarray):
            dtype = self.get_jax_dtype(value.dtype)
            with np.errstate(all="ignore"):  # ghost cells might have arbitrary values
                return jax.numpy.asarray(value, dtype=dtype, device=self.device)  # type: ignore
        msg = f"Unsupported type `{type(value).__name__}"
        raise TypeError(msg)

    def native_to_numpy(self, value: Any) -> Any:
        """Convert native values to numpy representation."""
        if isinstance(value, jax.Array):
            return np.asarray(value)
        return value

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        if not self.config["compile"]:
            return func

        return jax.jit(func)  # type: ignore

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

    def _make_local_ghost_cell_setter(self, bc: BCBase) -> JaxGhostCellSetter:
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
        from ._boundaries import make_virtual_point_evaluator

        normal = bc.normal
        axis = bc.axis
        rank = bc.rank

        # get information of the virtual points (ghost cells)
        vp_idx = bc.grid.shape[bc.axis] + 1 if bc.upper else 0
        np_idx = bc._get_value_cell_index(with_ghost_cells=False)
        vp_value = make_virtual_point_evaluator(bc, backend=self)

        # determine shape of data arrays
        data_full_shape = (bc.grid.dim,) * rank + bc.grid._shape_full
        if normal:
            # this has not been tested
            value_shape = (bc.grid.dim,) * max(rank - 1, 0) + tuple(
                bc.grid.shape[i] for i in range(bc.grid.num_axes) if i != axis
            )
        else:
            value_shape = (bc.grid.dim,) * rank + tuple(
                bc.grid.shape[i] for i in range(bc.grid.num_axes) if i != axis
            )

        if bc.grid.num_axes == 1:  # 1d grid

            def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                """Helper function setting the conditions on all axes."""
                assert data_full.shape == data_full_shape
                data_valid = data_full[..., 1:-1]
                val = vp_value(data_valid, (np_idx,), args=args)
                if normal:
                    return data_full.at[..., axis, vp_idx].set(val)
                return data_full.at[..., vp_idx].set(val)

        elif bc.grid.num_axes == 2:  # 2d grid
            if axis == 0:

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    assert data_full.shape == data_full_shape
                    data_valid = data_full[..., 1:-1, 1:-1]
                    val = vp_value(data_valid, (np_idx, slice(None)), args=args)
                    assert val.shape == value_shape
                    if normal:
                        return data_full.at[..., axis, vp_idx, 1:-1].set(val)
                    return data_full.at[..., vp_idx, 1:-1].set(val)

            elif axis == 1:

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    assert data_full.shape == data_full_shape
                    data_valid = data_full[..., 1:-1, 1:-1]
                    val = vp_value(data_valid, (slice(None), np_idx), args=args)
                    assert val.shape == value_shape
                    if normal:
                        return data_full.at[..., axis, 1:-1, vp_idx].set(val)
                    return data_full.at[..., 1:-1, vp_idx].set(val)

        elif bc.grid.num_axes == 3:  # 3d grid
            if axis == 0:

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    assert data_full.shape == data_full_shape
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    val = vp_value(
                        data_valid, (np_idx, slice(None), slice(None)), args=args
                    )
                    assert val.shape == value_shape
                    if normal:
                        return data_full.at[..., axis, vp_idx, 1:-1, 1:-1].set(val)
                    return data_full.at[..., vp_idx, 1:-1, 1:-1].set(val)

            elif axis == 1:

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    assert data_full.shape == data_full_shape
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    val = vp_value(
                        data_valid, (slice(None), np_idx, slice(None)), args=args
                    )
                    assert val.shape == value_shape
                    if normal:
                        return data_full.at[..., axis, 1:-1, vp_idx, 1:-1].set(val)
                    return data_full.at[..., 1:-1, vp_idx, 1:-1].set(val)

            elif axis == 2:

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    assert data_full.shape == data_full_shape
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    val = vp_value(
                        data_valid, (slice(None), slice(None), np_idx), args=args
                    )
                    assert val.shape == value_shape
                    if normal:
                        return data_full.at[..., axis, 1:-1, 1:-1, vp_idx].set(val)
                    return data_full.at[..., 1:-1, 1:-1, vp_idx].set(val)

        else:
            msg = "Too many axes"
            raise NotImplementedError(msg)

        return ghost_cell_setter

    def make_data_setter(  # type: ignore
        self, grid: GridBase, rank: int, bcs: BoundariesBase | None = None
    ) -> JaxDataSetter:
        """Create a function to set the valid part of a full data array.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                The grid for which the data setter is created
            rank (int):
                Rank of the data represented on the grid
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                Defines the boundary conditions for a particular grid, for which the
                setter should be defined.

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        num_axes = grid.num_axes
        shape_in_valid = (grid.dim,) * rank + grid.shape
        shape_in_full = (grid.dim,) * rank + grid._shape_full

        def get_full_data(data_valid: jax.Array, args=None) -> jax.Array:
            """Set valid part of the data (without ghost cells)

            Args:
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to the full array
                args:
                    Additional arguments (not used in this function)
            """
            assert data_valid.shape == shape_in_valid
            data_full = jnp.empty(shape_in_full, dtype=data_valid.dtype)
            if num_axes == 1:
                return data_full.at[..., 1:-1].set(data_valid)
            if num_axes == 2:
                return data_full.at[..., 1:-1, 1:-1].set(data_valid)
            if num_axes == 3:
                return data_full.at[..., 1:-1, 1:-1, 1:-1].set(data_valid)
            raise NotImplementedError

        if bcs is None:
            # just set the valid elements and leave ghost cells with arbitrary values
            return get_full_data

        # get the boundary conditions object
        bcs = grid.get_boundary_conditions(bcs, rank=rank)

        if not isinstance(bcs, BoundariesList):
            raise NotImplementedError

        # set the valid elements and the ghost cells according to boundary condition
        ghost_cell_setters = [
            self._make_local_ghost_cell_setter(bc_local)
            for bc_axis in bcs
            for bc_local in bc_axis
        ]

        def get_full_with_bcs(data_valid: jax.Array, args=None) -> jax.Array:
            """Set valid part of the data and the ghost cells using BCs.

            Args:
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to the full array
                args (dict):
                    Extra arguments affecting the boundary conditions
            """
            data_full = get_full_data(data_valid)
            for setter in ghost_cell_setters:
                data_full = setter(data_full, args=args)
            assert data_full.shape == shape_in_full
            return data_full

        return get_full_with_bcs

    def make_integrator(self, grid: GridBase) -> Callable[[jax.Array], jax.Array]:
        """Return function that integrates discretized data over a grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the integrator is defined

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        spatial_dims = tuple(range(-grid.num_axes, 0))
        cell_volumes = self.numpy_to_native(
            np.broadcast_to(grid.cell_volumes, grid.shape).astype(np.float64)
        )

        @self.compile_function
        def integrate_jax(arr: jax.Array) -> jax.Array:
            """Integrate data using cell volumes."""
            return jnp.sum(arr * cell_volumes, axis=spatial_dims)

        return integrate_jax

    def make_operator_no_bc(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        dtype: DTypeLike | None = None,
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
        dtype = self.get_jax_dtype(dtype or np.double)

        # create an operator with or without BCs
        jax_operator = operator_info.factory(grid, **kwargs)

        # compile the function and move it to the device
        return self.compile_function(jax_operator)

    @cached_method()
    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        dtype: DTypeLike | None = None,
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
            dtype (numpy dtype):
                The data type of the field.
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
        parameters, like time. When this backend is used together with JAX'
        just-in-time compilation (e.g. via :func:`jax.jit`), the values passed
        through `args` need to be compatible with JAX's JIT tracing rules.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        operator_raw = operator_info.factory(grid, **kwargs)

        # set the valid data
        get_full_with_bcs = self.make_data_setter(
            grid=grid, rank=operator_info.rank_in, bcs=bcs
        )

        @self.compile_function
        def apply_op_jax(
            arr: jax.Array,
            out: jax.Array | None = None,
            args: dict[str, Any] | None = None,
        ) -> jax.Array:
            """Set boundary conditions and apply operator."""
            if out is not None:
                msg = "`jax` arrays are immutable and cannot use `out`"
                raise RuntimeError(msg)
            # set boundary conditions
            arr_full = get_full_with_bcs(arr, args=args)
            # apply operator
            return operator_raw(arr_full)  # type: ignore

        return apply_op_jax  # type: ignore

    def make_inner_prod_operator(
        self, field: DataFieldBase, *, conjugate: bool = True
    ) -> Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]:
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

        def dot(a: jax.Array, b: jax.Array, out: jax.Array | None = None) -> jax.Array:
            """Jax implementation to calculate dot product between two fields."""
            rank_a = a.ndim - num_axes
            rank_b = b.ndim - num_axes
            if rank_a < 1 or rank_b < 1:
                msg = "Fields in dot product must have rank >= 1"
                raise TypeError(msg)
            if a.shape[rank_a:] != b.shape[rank_b:]:
                msg = "Shapes of fields are not compatible for dot product"
                raise ValueError(msg)
            if out is not None:
                msg = "jax implementation of inner product does not allow `out` arg."
                raise TypeError(msg)

            if conjugate:
                b = b.conj()

            if rank_a == 1 and rank_b == 1:  # result is scalar field
                return jnp.einsum("i...,i...->...", a, b)

            if rank_a == 2 and rank_b == 1:  # result is vector field
                return jnp.einsum("ij...,j...->i...", a, b)

            if rank_a == 1 and rank_b == 2:  # result is vector field
                return jnp.einsum("i...,ij...->j...", a, b)

            if rank_a == 2 and rank_b == 2:  # result is tensor-2 field
                return jnp.einsum("ij...,jk...->ik...", a, b)

            msg = f"Unsupported shapes ({a.shape}, {b.shape})"
            raise TypeError(msg)

        return dot

    def make_outer_prod_operator(
        self, field: DataFieldBase
    ) -> Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]:
        """Return operator calculating the outer product between two fields.

        This typically only supports products between two vector fields.

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
            a: jax.Array, b: jax.Array, out: jax.Array | None = None
        ) -> jax.Array:
            """Calculate the outer product using jax."""
            if out is not None:
                msg = "jax implementation of outer product does not allow `out` arg."
                raise TypeError(msg)
            return jnp.einsum("i...,j...->ij...", a, b)

        return outer

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
        from sympy.printing.numpy import JaxPrinter

        # collect all the user functions
        user_functions = expression.user_funcs.copy()
        if user_funcs is not None:
            user_functions.update(user_funcs)

        user_functions = {
            k: self.compile_function(v) for k, v in user_functions.items()
        }

        # initialize the printer that deals with numpy arrays correctly

        class ListArrayPrinter(JaxPrinter):
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
            modules=[user_functions, "jax"],
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
        return self.compile_function(result)

    def make_pde_rhs(
        self, eq: PDEBase, state: TField
    ) -> Callable[[jax.Array, float], jax.Array]:
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
        return self.compile_function(rhs_native)

    def make_gaussian_noise(
        self, field: TField, *, rng: np.random.Generator
    ) -> Callable[[], jax.Array]:
        """Create a function generating Gaussian white noise.

        This noise is already scaled to respect different cell volumes of the grid.

        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used to initialize the seed.
        """
        from jax import random as jrandom
        from jax.experimental.random import stateful_rng

        data_shape: tuple[int, ...] = field.data.shape
        scale = np.sqrt(1 / field.grid.cell_volumes)

        # initialize jax random number generator with numpy random number
        seed = rng.integers(0, np.iinfo(np.uint32).max + 1)
        jax_rng = stateful_rng(seed)

        @self.compile_function
        def gaussian_noise() -> jax.Array:
            """Helper function returning a noise realization."""
            key = jax_rng.key()
            return scale * jrandom.normal(key, shape=data_shape)  # type: ignore

        return gaussian_noise

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

        # create the backend-level stepping function
        inner_stepper = make_inner_stepper(solver, state)

        def stepper(state: TField, t_start: float, t_end: float) -> float:
            """Advance `state` by executing the backend-level stepping function."""
            # push state data to native backend
            state_tensor: jax.Array = solver.backend.numpy_to_native(state.data)
            # execute the backend-level stepping function
            state_tensor, t_last = inner_stepper(state_tensor, t_start, t_end)
            # retrieve data from native backend
            state.data[:] = solver.backend.native_to_numpy(state_tensor)
            return t_last

        return stepper  # type: ignore
