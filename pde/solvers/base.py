"""
Package that contains base classes for solvers

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type  # @UnusedImport

import numba as nb
import numpy as np

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.misc import classproperty


class SolverBase(metaclass=ABCMeta):
    """base class for solvers"""

    _subclasses: Dict[str, Type[SolverBase]] = {}  # all inheriting classes

    def __init__(self, pde: PDEBase):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
        """
        self.pde = pde
        self.info: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "pde_class": self.pde.__class__.__name__,
        }
        self._logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls
        if hasattr(cls, "name") and cls.name:
            if cls.name in cls._subclasses:
                logging.warning(f"Solver with name {cls.name} is already registered")
            cls._subclasses[cls.name] = cls

    @classmethod
    def from_name(cls, name: str, pde: PDEBase, **kwargs) -> SolverBase:
        r"""create solver class based on its name

        Solver classes are automatically registered when they inherit from
        :class:`SolverBase`. Note that this also requires that the respective python
        module containing the solver has been loaded before it is attempted to be used.

        Args:
            name (str):
                The name of the solver to construct
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            \**kwargs:
                Additional arguments for the constructor of the solver

        Returns:
            An instance of a subclass of :class:`SolverBase`
        """
        try:
            # obtain the solver class associated with `name`
            solver_class = cls._subclasses[name]
        except KeyError:
            # solver was not registered
            solvers = (
                f"'{solver}'"
                for solver in sorted(cls._subclasses.keys())
                if not solver.endswith("Solver")
            )
            raise ValueError(
                f"Unknown solver method '{name}'. Registered solvers are "
                + ", ".join(solvers)
            )

        return solver_class(pde, **kwargs)

    @classproperty
    def registered_solvers(cls) -> List[str]:  # @NoSelf
        """list of str: the names of the registered solvers"""
        return list(sorted(cls._subclasses.keys()))

    def _make_pde_rhs(
        self, state: FieldBase, backend: str = "auto"
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """obtain a function for evaluating the right hand side

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.

        Raises:
            RuntimeError: when a stochastic partial differential equation is encountered
            but `allow_stochastic == False`.

        Returns:
            A function that is called with data given by a :class:`~numpy.ndarray` and a
            time. The function returns the deterministic evolution rate and (if
            applicable) a realization of the associated noise.
        """
        if self.pde.is_sde:
            raise RuntimeError(
                f"Cannot create a deterministic stepper for a stochastic equation"
            )

        rhs = self.pde.make_pde_rhs(state, backend=backend)

        if hasattr(rhs, "_backend"):
            self.info["backend"] = rhs._backend  # type: ignore
        elif isinstance(rhs, nb.dispatcher.Dispatcher):
            self.info["backend"] = "numba"
        else:
            self.info["backend"] = "undetermined"

        return rhs

    def _make_sde_rhs(
        self, state: FieldBase, backend: str = "auto"
    ) -> Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
        """obtain a function for evaluating the right hand side

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.

        Raises:
            RuntimeError: when a stochastic partial differential equation is encountered
            but `allow_stochastic == False`.

        Returns:
            A function that is called with data given by a :class:`~numpy.ndarray` and a
            time. The function returns the deterministic evolution rate and (if
            applicable) a realization of the associated noise.
        """
        rhs = self.pde.make_sde_rhs(state, backend=backend)

        if hasattr(rhs, "_backend"):
            self.info["backend"] = rhs._backend  # type: ignore
        elif isinstance(rhs, nb.dispatcher.Dispatcher):
            self.info["backend"] = "numba"
        else:
            self.info["backend"] = "undetermined"

        return rhs

    @abstractmethod
    def make_stepper(
        self, state, dt: float = None
    ) -> Callable[[FieldBase, float, float], float]:
        pass
