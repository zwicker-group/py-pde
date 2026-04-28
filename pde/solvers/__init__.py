"""Solver classes define the strategy for evolving PDE states in time.

A solver object stores the numerical method and configuration and is used to construct
an executable stepping function that advances the current state.

.. autosummary::
   :nosignatures:

   ~adams_bashforth.AdamsBashforthSolver
   ~base.AdaptiveSolverBase
   ~base.SolverBase
   ~controller.Controller
   ~crank_nicolson.CrankNicolsonSolver
   ~euler.EulerSolver
   ~explicit_mpi.ExplicitMPISolver
   ~implicit.ImplicitSolver
   ~milstein.MilsteinSolver
   ~runge_kutta.RungeKuttaSolver
   ~scipy.ScipySolver
   ~base.registered_solvers


Inheritance structure of the classes:


.. inheritance-diagram::
        adams_bashforth.AdamsBashforthSolver
        crank_nicolson.CrankNicolsonSolver
        euler.EulerSolver
        explicit_mpi.ExplicitMPISolver
        implicit.ImplicitSolver
        milstein.MilsteinSolver
        runge_kutta.RungeKuttaSolver
        scipy.ScipySolver
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .adams_bashforth import AdamsBashforthSolver  # noqa: F401
from .base import *  # noqa: F403
from .controller import Controller  # noqa: F401
from .crank_nicolson import CrankNicolsonSolver  # noqa: F401
from .euler import EulerSolver, ExplicitSolver  # noqa: F401
from .implicit import ImplicitSolver  # noqa: F401
from .milstein import MilsteinSolver  # noqa: F401
from .runge_kutta import RungeKuttaSolver  # noqa: F401
from .scipy import ScipySolver  # noqa: F401

try:
    from .explicit_mpi import ExplicitMPISolver
except ImportError:
    # MPI modules do not seem to be properly available
    ExplicitMPISolver = None  # type: ignore
