"""
Solvers define how a PDE is solved, i.e., how the initial state is advanced in time.

.. autosummary::
   :nosignatures:

   ~controller.Controller
   ~explicit.ExplicitSolver
   ~explicit_mpi.ExplicitMPISolver
   ~implicit.ImplicitSolver
   ~scipy.ScipySolver
   ~registered_solvers
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import List

from .controller import Controller
from .explicit import ExplicitSolver
from .implicit import ImplicitSolver
from .scipy import ScipySolver

try:
    from .explicit_mpi import ExplicitMPISolver
except ImportError:
    # MPI modules do not seem to be properly available
    ExplicitMPISolver = None  # type: ignore


def registered_solvers() -> List[str]:
    """returns all solvers that are currently registered

    Returns:
        list of str: List with the names of the solvers
    """
    from .base import SolverBase

    return SolverBase.registered_solvers  # type: ignore


__all__ = [
    "Controller",
    "ExplicitSolver",
    "ImplicitSolver",
    "ScipySolver",
    "registered_solvers",
]
