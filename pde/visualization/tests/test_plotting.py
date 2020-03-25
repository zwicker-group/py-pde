'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import os
import tempfile

from .. import plotting
from ...grids import UnitGrid
from ...fields import ScalarField, FieldCollection
from ...storage import get_memory_storage
from ...tools.misc import skipUnlessModule



@skipUnlessModule("matplotlib")
def test_scalar_field_plot():
    """ test ScalarFieldPlot class"""
    # create some data
    state = ScalarField.random_uniform(UnitGrid([16, 16]))
    for scale in [(0, 1), 1, 'automatic', 'symmetric', 'unity']:
        sfp = plotting.ScalarFieldPlot(state, scale=scale)
        with tempfile.NamedTemporaryFile(suffix='.png') as fp:
            sfp.savefig(fp.name)
            assert os.stat(fp.name).st_size > 0

    sfp = plotting.ScalarFieldPlot(state, quantities={'source': None})
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        sfp.savefig(fp.name)
        assert os.stat(fp.name).st_size > 0



@skipUnlessModule("matplotlib")
def test_scalar_plot():
    """ test Simple simulation """
    # create some data
    state = ScalarField.random_uniform(UnitGrid([16, 16]), label='test')
    with get_memory_storage(state) as storage:
        storage.append(state.data, 0)
        storage.append(state.data, 1)
    
    # check creating an overview image
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        plotting.plot_magnitudes(storage, filename=fp.name)
        assert os.stat(fp.name).st_size > 0

    # check creating an kymograph
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        plotting.plot_kymograph(storage, filename=fp.name)
        assert os.stat(fp.name).st_size > 0



@skipUnlessModule("matplotlib")
def test_collection_plot():
    """ test Simple simulation """
    # create some data
    field = FieldCollection([ScalarField(UnitGrid([8, 8]), label='first'),
                             ScalarField(UnitGrid([8, 8]))])
    with get_memory_storage(field) as storage:
        storage.append(field.data)
    
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        plotting.plot_magnitudes(storage, filename=fp.name)
        assert os.stat(fp.name).st_size > 0

