"""Solvers define how a PDE is solved, i.e., how the initial state is advanced in time.

.. autosummary::
   :nosignatures:

   ~controller.Controller
   ~explicit.EulerSolver
   ~explicit.RungeKuttaSolver
   ~explicit_mpi.ExplicitMPISolver
   ~implicit.ImplicitSolver
   ~crank_nicolson.CrankNicolsonSolver
   ~adams_bashforth.AdamsBashforthSolver
   ~scipy.ScipySolver
   ~registered_solvers


Inheritance structure of the classes:


.. inheritance-diagram::
        adams_bashforth.AdamsBashforthSolver
        crank_nicolson.CrankNicolsonSolver
        explicit.EulerSolver
        explicit.RungeKuttaSolver
        implicit.ImplicitSolver
        scipy.ScipySolver
        explicit_mpi.ExplicitMPISolver
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .adams_bashforth import AdamsBashforthSolver  # noqa: F401
from .base import *  # noqa: F403
from .controller import Controller  # noqa: F401
from .crank_nicolson import CrankNicolsonSolver  # noqa: F401
from .explicit import EulerSolver, ExplicitSolver, RungeKuttaSolver  # noqa: F401
from .implicit import ImplicitSolver  # noqa: F401
from .scipy import ScipySolver  # noqa: F401

try:
    from .explicit_mpi import ExplicitMPISolver
except ImportError:
    # MPI modules do not seem to be properly available
    ExplicitMPISolver = None  # type: ignore
