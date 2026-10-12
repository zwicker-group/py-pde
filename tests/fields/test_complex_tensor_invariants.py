"""Check tensor invariants retain complex values."""

import itertools

import numpy as np
import pytest

from pde import Tensor2Field, UnitGrid


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["invariant2", "det", "determinant", "invariant3"])
def test_tensor_invariants_complex_values(dim, dtype, kind):
    """Compare against trace identities and a scalar determinant expansion."""
    grid = UnitGrid([2] * dim)
    matrix = np.arange(dim * dim).reshape(dim, dim) / 5 + (3 + 2j) * np.eye(dim)
    if not np.issubdtype(dtype, np.complexfloating):
        matrix = matrix.real
    scale = 1 + grid.cell_coords.sum(axis=-1) / 10
    data = matrix[(...,) + (None,) * dim] * scale
    field = Tensor2Field(grid, data, dtype=dtype)
    original = field.data.copy()
    expected = np.empty(grid.shape, dtype=np.complex128)
    for index in np.ndindex(grid.shape):
        point = field.data[(..., *index)].astype(np.complex128)
        if kind == "invariant2":
            # Preserve the package's established half-scaled convention.
            expected[index] = (np.trace(point) ** 2 - np.trace(point @ point)) / 4
        else:
            determinant = 0j
            for permutation in itertools.permutations(range(dim)):
                inversions = sum(
                    permutation[i] > permutation[j]
                    for i in range(dim)
                    for j in range(i + 1, dim)
                )
                term = (-1) ** inversions
                for i, j in enumerate(permutation):
                    term *= point[i, j]
                determinant += term
            expected[index] = determinant
    result = field.to_scalar(kind)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(field.data, original)
