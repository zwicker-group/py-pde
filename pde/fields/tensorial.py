"""
Defines a tensorial field of rank 2 over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numba as nb
import numpy as np

from ..tools.docstrings import fill_in_docstring
from ..tools.misc import get_common_dtype
from ..tools.numba import get_common_numba_dtype, jit
from .base import DataFieldBase
from .scalar import ScalarField
from .vectorial import VectorField

if TYPE_CHECKING:
    from ..grids.boundaries.axes import BoundariesData  # @UnusedImport


class Tensor2Field(DataFieldBase):
    """Single tensor field of rank 2 on a grid

    Attributes:
        grid (:class:`~pde.grids.GridBase`):
            The underlying grid defining the discretization
        data (:class:`numpy.ndarray`):
            Tensor components at the support points of the grid
        label (str):
            Name of the field
    """

    rank = 2

    def __getitem__(self, key: Tuple[int, int]) -> ScalarField:
        """ extract a component of the VectorField """
        try:
            if len(key) != 2:
                raise IndexError("Index must be given as two integers")
        except TypeError:
            raise IndexError("Index must be given as two integers")
        return ScalarField(self.grid, self.data[key])

    @DataFieldBase._data_flat.setter  # type: ignore
    def _data_flat(self, value):
        """ set the data from a value from a collection """
        dim = self.grid.dim
        self._data = value.reshape(dim, dim, *self.grid.shape)
        # check whether both point to the same memory location
        addr_value = value.__array_interface__["data"][0]
        addr_self_data = self._data.__array_interface__["data"][0]
        assert addr_value == addr_self_data

    def dot(
        self,
        other: Union[VectorField, "Tensor2Field"],
        out: Optional[Union[VectorField, "Tensor2Field"]] = None,
        *,
        conjugate: bool = True,
        label: str = "dot product",
    ) -> Union[VectorField, "Tensor2Field"]:
        """calculate the dot product involving a tensor field

        This supports the dot product between two tensor fields as well as the
        product between a tensor and a vector. The resulting fields will be a
        tensor or vector, respectively.

        Args:
            other (VectorField or Tensor2Field):
                the second field
            out (VectorField or Tensor2Field, optional):
                Optional field to which the  result is written.
            conjugate (bool):
                Whether to use the complex conjugate for the second operand
            label (str, optional):
                Name of the returned field

        Returns:
            VectorField or Tensor2Field: the result of applying the dot operator
        """
        # check input
        self.grid.assert_grid_compatible(other.grid)
        if not isinstance(other, (VectorField, Tensor2Field)):
            raise TypeError("Second term must be a vector or tensor field")

        # create and check the output instance
        if out is None:
            out = other.__class__(self.grid, dtype=get_common_dtype(self, other))
        else:
            assert isinstance(out, other.__class__)
            self.grid.assert_grid_compatible(out.grid)

        # calculate the result
        other_data = other.data.conjugate() if conjugate else other.data
        np.einsum("ij...,j...->i...", self.data, other_data, out=out.data)
        if label is not None:
            out.label = label

        return out

    __matmul__ = dot  # support python @-syntax for matrix multiplication

    def make_dot_operator(self, conjugate: bool = True) -> Callable:
        """return operator calculating the dot product involving vector fields

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Warning:
            This function does not check types or dimensions.

        Args:
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`numpy.ndarray`, which
            contain the discretized data of the two operands. An optional third
            argument can specify the output array to which the result is
            written. Note that the returned function is jitted with numba for
            speed.
        """
        dim = self.grid.dim

        # create the inner function calculating the dot product
        if conjugate:

            @jit
            def inner(a, b, out):
                """ calculate dot product between fields `a` and `b` """
                for i in range(dim):
                    out[i] = a[i, 0] * b[0].conjugate()  # overwrite data in out
                    for j in range(1, dim):
                        out[i] += a[i, j] * b[j].conjugate()
                return out

        else:

            @jit
            def inner(a, b, out):
                """ calculate dot product between fields `a` and `b` """
                for i in range(dim):
                    out[i] = a[i, 0] * b[0]  # overwrite potential data in out
                    for j in range(1, dim):
                        out[i] += a[i, j] * b[j]
                return out

        # build the outer function with the correct signature
        if nb.config.DISABLE_JIT:

            def dot(a, b, out=None):
                """wrapper deciding whether the underlying function is called
                with or without `out`."""
                if out is None:
                    out = np.empty(b.shape, dtype=get_common_dtype(a, b))
                return inner(a, b, out)

        else:

            @nb.generated_jit
            def dot(a, b, out=None):
                """wrapper deciding whether the underlying function is called
                with or without `out`."""
                if isinstance(a, nb.types.Number):
                    # simple scalar call -> do not need to allocate anything
                    raise RuntimeError("Dot needs to be called with fields")

                elif isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                    # function is called without `out`
                    dtype = get_common_numba_dtype(a, b)

                    def f_with_allocated_out(a, b, out):
                        """ helper function allocating output array """
                        out = np.empty(b.shape, dtype=dtype)
                        return inner(a, b, out=out)

                    return f_with_allocated_out

                else:
                    # function is called with `out` argument
                    return inner

        return dot

    def get_dot_operator(self) -> Callable:
        """return operator calculating the dot product involving vector fields

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Warning:
            This function does not check types or dimensions.

        Returns:
            function that takes two instance of :class:`numpy.ndarray`, which
            contain the discretized data of the two operands. An optional third
            argument can specify the output array to which the result is
            written. Note that the returned function is jitted with numba for
            speed.
        """
        # Deprecated this method on 2020-10-07
        import warnings

        warnings.warn(
            "get_dot_operator() method is deprecated. Use the `make_dot_operator()` "
            "method instead",
            DeprecationWarning,
        )
        return self.make_dot_operator()

    @fill_in_docstring
    def divergence(
        self,
        bc: "BoundariesData",
        out: Optional[VectorField] = None,
        *,
        label: str = "divergence",
    ) -> VectorField:
        """apply (tensor) divergence and return result as a field

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            out (VectorField, optional):
                Optional scalar field to which the  result is written.
            label (str, optional):
                Name of the returned field

        Returns:
            VectorField: the result of applying the operator
        """
        tensor_divergence = self.grid.get_operator("tensor_divergence", bc=bc)
        if out is None:
            out = VectorField(self.grid, tensor_divergence(self.data), label=label)
        else:
            assert isinstance(out, VectorField)
            tensor_divergence(self.data, out=out.data)
        return out

    @property
    def integral(self):
        """ :class:`numpy.ndarray`: integral of each component over space """
        return self.grid.integrate(self.data)

    def transpose(self, label: str = "transpose") -> "Tensor2Field":
        """return the transpose of the tensor field

        Args:
            label (str, optional): Name of the returned field

        Returns:
            Tensor2Field: holding the transpose of the tensor field
        """
        axes = (1, 0) + tuple(range(2, 2 + self.grid.dim))
        return Tensor2Field(self.grid, self.data.transpose(axes), label=label)

    def symmetrize(
        self, make_traceless: bool = False, inplace: bool = False
    ) -> "Tensor2Field":
        """symmetrize the tensor field in place

        Args:
            make_traceless (bool):
                Determines whether the result is also traceless
            inplace (bool):
                Flag determining whether to symmetrize the current field or
                return a new one
        """
        if inplace:
            out = self
        else:
            out = self.copy()

        out += self.transpose()
        out *= 0.5

        if make_traceless:
            dim = self.grid.dim
            value = self.trace() / dim
            for i in range(dim):
                out._data[i, i] -= value.data
        return out

    def to_scalar(
        self, scalar: str = "auto", *, label: Optional[str] = "scalar `{scalar}`"
    ) -> ScalarField:
        r""" return a scalar field by applying `method`
        
        The invariants of the tensor field :math:`\boldsymbol{A}` are
        
        .. math::
            I_1 &= \mathrm{tr}(\boldsymbol{A}) \\
            I_2 &= \frac12 \left[
                (\mathrm{tr}(\boldsymbol{A})^2 -
                \mathrm{tr}(\boldsymbol{A}^2)
            \right] \\
            I_3 &= \det(A)
            
        where `tr` denotes the trace and `det` denotes the determinant.
        Note that the three invariants can only be distinct and non-zero in three
        dimensions. In two dimensional spaces, we have the identity :math:`2 I_2 = I_3`
        and in one-dimensional spaces, we have :math:`I_1 = I_3` as well as
        :math:`I_2 = 0`.
            
        Args:
            scalar (str):
                The method to calculate the scalar. Possible choices include `norm` (the
                default), `min`, `max`, `squared_sum`, `norm_squared`, `trace` (or
                `invariant1`), `invariant2`, and `determinant` (or `invariant3`)
            label (str, optional):
                Name of the returned field
            
        Returns:
            :class:`pde.fields.scalar.ScalarField`: the scalar field after
            applying the operation
        """
        if scalar == "auto":
            scalar = "norm"

        if scalar == "norm":
            data = np.linalg.norm(self.data, axis=(0, 1))

        elif scalar == "min":
            data = np.min(self.data, axis=(0, 1))

        elif scalar == "max":
            data = np.max(self.data, axis=(0, 1))

        elif scalar == "squared_sum":
            data = np.sum(self.data ** 2, axis=(0, 1))

        elif scalar == "norm_squared":
            data = np.sum(self.data * self.data.conjugate(), axis=(0, 1))

        elif scalar == "trace" or scalar == "invariant1":
            data = self.data.trace(axis1=0, axis2=1)

        elif scalar == "invariant2":
            data = np.zeros(self.grid.shape)
            for i in range(self.grid.dim):
                for j in range(i):
                    data += (
                        self.data[i, i] * self.data[j, j]
                        - self.data[i, j] * self.data[j, i]
                    )
            data *= 0.5

        elif scalar in {"det", "determinant", "invariant3"}:
            if self.grid.dim == 1:
                data = self.data[0, 0]
            else:
                data = np.zeros(self.grid.shape)
                # this iterates over all of space and might thus be slow, but
                # the interface of np.linalg.det is not very flexible. We could
                # in principle use the definition of np.linalg.det without the
                # multiple checks to gain some speed
                for i in np.ndindex(self.grid.shape):
                    data[i] = np.linalg.det(self.data[(...,) + i])  # type: ignore

        else:
            raise ValueError(
                f"Unknown method `{scalar}` for `to_scalar`. Valid methods are `norm`, "
                "`min`, `max`, squared_sum`, `norm_squared`, `trace`, `determinant`, "
                "and `invariant#`, where # is 1, 2, or 3"
            )

        # determine label of the result
        if self.label is None:
            if label is not None:
                label = label.format(scalar=scalar)
        else:
            label = f"{scalar} of {self.label}"

        return ScalarField(self.grid, data, label=label)

    def trace(self, label: Optional[str] = "trace") -> ScalarField:
        """return the trace of the tensor field as a scalar field

        Args:
            label (str, optional): Name of the returned field

        Returns:
            ScalarField: holding the trace
        """
        return self.to_scalar(scalar="trace", label=label)
