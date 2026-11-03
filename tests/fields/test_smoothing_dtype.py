"""Check field dtype preservation when smoothing allocates an output."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pde import CartesianGrid, ScalarField, Tensor2Field, VectorField


@pytest.mark.parametrize("field_type", [ScalarField, VectorField, Tensor2Field])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("periodic", [(True, True), (False, False), (True, False)])
def test_smoothing_preserves_dtype(field_type, dtype, periodic):
    """Allocation preserves values, dtype and component axes."""
    grid = CartesianGrid([[0, 2], [0, 3]], [7, 9], periodic=periodic)
    rng = np.random.default_rng(0)
    field = field_type(grid, dtype=dtype, label="original")
    data = rng.normal(size=field.data.shape)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=field.data.shape)
    field.data = data
    original = field.data.copy()
    sigma = 0.3
    component_axes = field.rank
    expected = gaussian_filter(
        original,
        sigma=(0,) * component_axes + tuple(sigma / grid.discretization),
        mode=("nearest",) * component_axes
        + tuple("wrap" if p else "reflect" for p in periodic),
    )
    result = field.smooth(sigma)
    assert result.dtype == (field.dtype if field.is_complex else np.dtype(float))
    assert result.label == field.label
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(field.data, original)
    assert not np.shares_memory(result.data, field.data)

    explicit = field_type(grid, dtype=result.dtype)
    assert field.smooth(sigma, out=explicit) is explicit
    np.testing.assert_array_equal(result.data, explicit.data)
    assert field.smooth(sigma, out=field) is field
    np.testing.assert_allclose(result.data, field.data, rtol=1e-6, atol=1e-7)
