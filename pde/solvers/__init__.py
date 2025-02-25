"""Solvers define how a PDE is solved, i.e., how the initial state is advanced in time.

.. autosummary::
   :nosignatures:

   ~controller.Controller
   ~explicit.ExplicitSolver
   ~explicit_mpi.ExplicitMPISolver
   ~implicit.ImplicitSolver
   ~crank_nicolson.CrankNicolsonSolver
   ~adams_bashforth.AdamsBashforthSolver
   ~scipy.ScipySolver
   ~registered_solvers


Inheritance structure of the classes:


.. inheritance-diagram:: adams_bashforth.AdamsBashforthSolver
        crank_nicolson.CrankNicolsonSolver
        explicit.ExplicitSolver
        implicit.ImplicitSolver
        scipy.ScipySolver
        explicit_mpi.ExplicitMPISolver
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .adams_bashforth import AdamsBashforthSolver
from .controller import Controller
from .crank_nicolson import CrankNicolsonSolver
from .explicit import ExplicitSolver
from .implicit import ImplicitSolver
from .scipy import ScipySolver

try:
    from .explicit_mpi import ExplicitMPISolver
except ImportError:
    # MPI modules do not seem to be properly available
    ExplicitMPISolver = None  # type: ignore


def registered_solvers() -> list[str]:
    """Returns all solvers that are currently registered.

    Returns:
        list of str: List with the names of the solvers
    """
    from .base import SolverBase

    return SolverBase.registered_solvers  # type: ignore


__all__ = [
    "Controller",
    "ExplicitSolver",
    "ImplicitSolver",
    "CrankNicolsonSolver",
    "AdamsBashforthSolver",
    "ScipySolver",
    "registered_solvers",
]
