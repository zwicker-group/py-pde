'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from .. import movies
from ...grids import UnitGrid
from ...fields import ScalarField
from ...storage import MemoryStorage
from ...pdes import DiffusionPDE
from ...tools.misc import skipUnlessModule



@skipUnlessModule("matplotlib")
def test_movie(tmp_path):
    """ test Movie class"""
    import matplotlib.pyplot as plt
    
    path = tmp_path / "test_movie.mov"
    
    try:
        with movies.Movie(path) as movie:
            # iterate over all time steps
            plt.plot([0, 1], [0, 1])
            movie.add_figure()
            movie.add_figure()
        
            # save movie
            movie.save()
    except RuntimeError:
        pass  # can happen when ffmpeg is not installed
    else:
        assert path.stat().st_size > 0



@skipUnlessModule("matplotlib")
def test_movie_scalar(tmp_path):
    """ test Movie class"""
    
    # create some data
    state = ScalarField.random_uniform(UnitGrid([16, 16]))
    pde = DiffusionPDE()
    storage = MemoryStorage()
    tracker = storage.tracker(interval=1)
    pde.solve(state, t_range=2, dt=1e-2, backend='numpy', tracker=tracker)
    
    # check creating the movie
    path = tmp_path / "test_movie.mov"
    
    try:
        movies.movie_scalar(storage, filename=path, progress=False)
    except RuntimeError:
        pass  # can happen when ffmpeg is not installed
    else:
        assert path.stat().st_size > 0
