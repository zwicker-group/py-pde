"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.fields import ScalarField
from pde.grids import UnitGrid
from pde.grids.mesh import GridMesh


@pytest.mark.parametrize("decomp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_split_fields(decomp):
    grid = UnitGrid([8, 8])
    mesh = GridMesh.from_grid(grid, decomp)

    field = ScalarField(grid)
    field._data_full = np.random.uniform(size=grid._shape_full)
    subfields = [mesh.extract_subfield(field, node_id) for node_id in range(len(mesh))]

    field_comb = mesh.combine_field_data([f.data for f in subfields])
    assert field == field_comb
