"""Check that integration axes address spatial axes independently of tensor rank."""

import numpy as np
import pytest

from pde import CartesianGrid, CylindricalSymGrid


@pytest.mark.parametrize(
    "grid",
    [
        CartesianGrid([[0, 2], [-1, 4]], [3, 4]),
        CylindricalSymGrid(2, [-1, 4], [3, 4]),
    ],
)
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("axes", [-1, (-2,), (-2, -1), (0, -1), ()])
def test_integration_negative_spatial_axes(grid, rank, axes):
    """Negative spatial axes must use the same weights and component offsets."""
    shape = (grid.dim,) * rank + grid.shape
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    positive = tuple(ax % grid.num_axes for ax in np.atleast_1d(axes))

    result = grid.integrate(data, axes=axes)

    np.testing.assert_allclose(result, grid.integrate(data, axes=positive))


@pytest.mark.parametrize("axes", [2, -3, (0, -2)])
def test_integration_invalid_axes(axes):
    """Component dimensions must not make invalid spatial axes acceptable."""
    grid = CartesianGrid([[0, 2], [-1, 4]], [3, 4])
    with pytest.raises((np.exceptions.AxisError, ValueError)):
        grid.integrate(np.ones((2, 3, 4)), axes=axes)
