"""Functions and classes for visualizing simulations.

.. autosummary::
   :nosignatures:

   ~movies.movie
   ~movies.movie_scalar
   ~movies.movie_multiple
   ~plotting.plot_interactive
   ~plotting.plot_kymograph
   ~plotting.plot_kymographs
   ~plotting.plot_magnitudes

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .movies import movie, movie_multiple, movie_scalar  # noqa: F401
from .plotting import (  # noqa: F401
    plot_interactive,
    plot_kymograph,
    plot_kymographs,
    plot_magnitudes,
)
