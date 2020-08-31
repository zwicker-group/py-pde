"""
Solvers define how a pde is solved, i.e., advanced in time.

.. autosummary::
   :nosignatures:

   ~controller.Controller
   ~explicit.ExplicitSolver
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


def registered_solvers() -> List[str]:
    """returns all solvers that are registered in the package

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
