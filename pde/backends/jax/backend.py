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

from ...tools.cache import cached_method
from ..numpy.backend import NumpyBackend

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ...grids import GridBase
    from ...grids.boundaries.axes import BoundariesBase
    from ...grids.boundaries.local import BCBase
    from ...tools.config import Config
    from ...tools.typing import (
        NumericArray,
        OperatorInfo,
        OperatorType,
    )
    from ..base import TFunc
    from .typing import JaxDataSetter, JaxGhostCellSetter, JaxOperatorImplType


class JaxBackend(NumpyBackend):
    """Defines :mod:`jax` backend."""

    implementation = "jax"

    _dtype_cache: dict[str, dict[DTypeLike, np.dtype]] = defaultdict(dict)
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

    def get_jax_dtype(self, dtype: DTypeLike) -> np.dtype:
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
        np_dtype: np.dtype = np.dtype(dtype)
        try:
            # try returning type from cache
            return type_cache[np_dtype]
        except KeyError:
            pass

        try:
            # Try to create a tensor of this dtype on the device
            jnp.empty(1, dtype=np_dtype, device=self.device)
        except TypeError:
            # dtype is not supported, so we see whether we need to use downcasting
            if self.config["dtype_downcasting"] and np_dtype == np.float64:
                if not self._emitted_downcast_warning:
                    self._logger.warning(
                        " %s device doesn't support float64, so we use float32 instead",
                        self.device.type,
                    )
                    self._emitted_downcast_warning = True
                jax_dtype: np.dtype = np.float32
            else:
                raise
        else:
            jax_dtype = np_dtype

        # store dtype in cache
        type_cache[np_dtype] = jax_dtype
        return jax_dtype

    def from_numpy(self, value: Any) -> Any:
        """Convert values from numpy to jax representation.

        This method also ensures that the value is copied to the selected device.
        """
        if isinstance(value, (jax.Array, numbers.Number)):
            return jax.device_put(value, self.device)
        if isinstance(value, np.ndarray):
            dtype = self.get_jax_dtype(value.dtype)
            with np.errstate(under="ignore", over="ignore"):
                return jax.numpy.asarray(value, dtype=dtype, device=self.device)
        msg = f"Unsupported type `{value.__type__}"
        raise TypeError(msg)

    def to_numpy(self, value: Any) -> Any:
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
        if self.config["compile"]:
            return jax.jit(func)  # type: ignore
        return func

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

        # get information of the virtual points (ghost cells)
        vp_idx = bc.grid.shape[bc.axis] + 1 if bc.upper else 0
        np_idx = bc._get_value_cell_index(with_ghost_cells=False)
        vp_value = make_virtual_point_evaluator(bc, backend=self)

        if bc.grid.num_axes == 1:  # 1d grid

            def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                """Helper function setting the conditions on all axes."""
                data_valid = data_full[..., 1:-1]
                val = vp_value(data_valid, (np_idx,), args=args)
                if normal:
                    return data_full.at[..., axis, vp_idx].set(val)
                return data_full.at[..., vp_idx].set(val)

        elif bc.grid.num_axes == 2:  # 2d grid
            # TODO: We might need to get rid of the loops in setting the boundaries
            if bc.axis == 0:
                num_y = bc.grid.shape[1]

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for j in range(num_y):
                        val = vp_value(data_valid, (np_idx, j), args=args)
                        if normal:
                            data_full = data_full.at[..., axis, vp_idx, j + 1].set(val)
                        else:
                            data_full = data_full.at[..., vp_idx, j + 1].set(val)
                    return data_full

            elif bc.axis == 1:
                num_x = bc.grid.shape[0]

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1]
                    for i in range(num_x):
                        val = vp_value(data_valid, (i, np_idx), args=args)
                        if normal:
                            data_full = data_full.at[..., axis, i + 1, vp_idx].set(val)
                        else:
                            data_full = data_full.at[..., i + 1, vp_idx].set(val)
                    return data_full

        elif bc.grid.num_axes == 3:  # 3d grid
            if bc.axis == 0:
                num_y, num_z = bc.grid.shape[1:]

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for j in range(num_y):
                        for k in range(num_z):
                            val = vp_value(data_valid, (np_idx, j, k), args=args)
                            if normal:
                                data_full = data_full.at[
                                    ..., axis, vp_idx, j + 1, k + 1
                                ].set(val)
                            else:
                                data_full = data_full.at[..., vp_idx, j + 1, k + 1].set(
                                    val
                                )
                    return data_full

            elif bc.axis == 1:
                num_x, num_z = bc.grid.shape[0], bc.grid.shape[2]

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for k in range(num_z):
                            val = vp_value(data_valid, (i, np_idx, k), args=args)
                            if normal:
                                data_full = data_full.at[
                                    ..., axis, i + 1, vp_idx, k + 1
                                ].set(val)
                            else:
                                data_full = data_full.at[..., i + 1, vp_idx, k + 1].set(
                                    val
                                )
                    return data_full

            elif bc.axis == 2:
                num_x, num_y = bc.grid.shape[:2]

                def ghost_cell_setter(data_full: jax.Array, args=None) -> jax.Array:
                    """Helper function setting the conditions on all axes."""
                    data_valid = data_full[..., 1:-1, 1:-1, 1:-1]
                    for i in range(num_x):
                        for j in range(num_y):
                            val = vp_value(data_valid, (i, j, np_idx), args=args)
                            if normal:
                                data_full = data_full.at[
                                    ..., axis, i + 1, j + 1, vp_idx
                                ].set(val)
                            else:
                                data_full = data_full.at[..., i + 1, j + 1, vp_idx].set(
                                    val
                                )
                    return data_full

        else:
            msg = "Too many axes"
            raise NotImplementedError(msg)

        return ghost_cell_setter  # type: ignore

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
        shape_in_full = (grid.dim,) * rank + grid._shape_full

        def get_full_data(data_valid: jax.Array, args=None) -> jax.Array:
            """Set valid part of the data (without ghost cells)

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The full array with ghost cells that the data is written to
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to `data_full`
                args:
                    Additional arguments (not used in this function)
            """
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
            return get_full_data  # type: ignore[return-value]

        # get the boundary conditions object
        bcs = grid.get_boundary_conditions(bcs)

        # set the valid elements and the ghost cells according to boundary condition
        ghost_cell_setters = [
            self._make_local_ghost_cell_setter(bc_local)
            for bc_axis in bcs
            for bc_local in bc_axis
        ]

        def get_full_with_bcs(data_valid: jax.Array, args=None) -> jax.Array:
            """Set valid part of the data and the ghost cells using BCs.

            Args:
                data_full (:class:`~numpy.ndarray`):
                    The full array with ghost cells that the data is written to
                data_valid (:class:`~numpy.ndarray`):
                    The valid data that is written to `data_full`
                args (dict):
                    Extra arguments affecting the boundary conditions
            """
            data_full = get_full_data(data_valid)
            for setter in ghost_cell_setters:
                data_full = setter(data_full, args=args)
            return data_full

        return get_full_with_bcs

    def make_operator_no_bc(  # type: ignore
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        dtype: DTypeLike | None = None,
        native: bool = False,
        **kwargs,
    ) -> JaxOperatorImplType:
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
        dtype = self.get_jax_dtype(dtype or np.double)

        # create an operator with or without BCs
        jax_operator = operator_info.factory(grid, **kwargs)

        # compile the function and move it to the device
        jax_operator_jitted = self.compile_function(jax_operator)

        if native:
            return jax_operator_jitted  # type: ignore

        def operator_no_bc(arr: NumericArray, out: NumericArray) -> None:
            arr_jax = self.from_numpy(arr)
            out_jax = jax_operator_jitted(arr_jax)  # type: ignore
            out[...] = self.to_numpy(out_jax)

        return operator_no_bc  # type: ignore

    @cached_method()
    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        dtype: DTypeLike | None = None,
        native: bool = False,
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
            native (bool):
                If True, the returned functions expects the native data representation
                of the backend. Otherwise, the input and output are expected to be
                :class:`~numpy.ndarray`.
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
        operator_raw = operator_info.factory(grid, **kwargs)

        # set the valid data
        get_full_with_bcs = self.make_data_setter(
            grid=grid, rank=operator_info.rank_out, bcs=bcs
        )

        @self.compile_function
        def apply_op_jax(arr: jax.Array, args=None) -> jax.Array:
            """Set boundary conditions and apply operator."""
            # set boundary conditions
            arr_full = get_full_with_bcs(arr, args=args)
            # apply operator
            return operator_raw(arr_full)

        if native:
            return apply_op_jax

        # calculate shapes of the full data
        shape_in_valid = (grid.dim,) * operator_info.rank_in + grid.shape
        shape_out = (grid.dim,) * operator_info.rank_out + grid.shape

        # define numpy version of the operator
        def apply_op(
            arr: NumericArray, out: NumericArray | None = None, args=None
        ) -> NumericArray:
            """Set boundary conditions and apply operator."""
            # check input array
            if arr.shape != shape_in_valid:
                msg = f"Incompatible shapes {arr.shape} != {shape_in_valid}"
                raise ValueError(msg)
            # ensure `out` array is allocated and has the right shape
            if out is not None and out.shape != shape_out:
                msg = f"Incompatible shapes {out.shape} != {shape_out}"
                raise ValueError(msg)

            # convert data to jax and apply operator
            arr_jax = self.from_numpy(arr)
            out_jax = apply_op_jax(arr_jax, args=args)

            # return result
            if out is None:
                out = self.to_numpy(out_jax)
            else:
                out[:] = self.to_numpy(out_jax)

            # return valid part of the output
            return out

        return apply_op  # type: ignore
