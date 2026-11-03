"""Test allocation of tensor fields for complex vector outer products."""

import numpy as np
import pytest

from pde import Tensor2Field, UnitGrid, VectorField


@pytest.mark.parametrize(
    "dtype_left", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "dtype_right", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_outer_product_complex_allocation(dtype_left, dtype_right):
    """The default output must represent products of both operands."""
    grid = UnitGrid([3, 4])
    data = np.arange(24).reshape(2, 3, 4) / 7 + 1
    left_data = data + 2j if np.issubdtype(dtype_left, np.complexfloating) else data
    right_data = (
        2 * data - 3j if np.issubdtype(dtype_right, np.complexfloating) else 2 * data
    )
    left = VectorField(grid, left_data, dtype=dtype_left)
    right = VectorField(grid, right_data, dtype=dtype_right)
    expected = left.data[:, None] * right.data[None, :]

    result = left.outer_product(right, label="dyad")

    np.testing.assert_allclose(result.data, expected, rtol=2e-7)
    assert result.grid == grid
    assert result.label == "dyad"
    assert result.is_complex == (left.is_complex or right.is_complex)
    explicit = Tensor2Field(grid, dtype=result.dtype)
    assert left.outer_product(right, out=explicit) is explicit
    np.testing.assert_array_equal(explicit.data, result.data)
    np.testing.assert_array_equal(left.data, left_data.astype(dtype_left))
    np.testing.assert_array_equal(right.data, right_data.astype(dtype_right))
