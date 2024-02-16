"""
Base class for defining partial differential equations

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import copy
import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

import numpy as np

from ..fields import FieldCollection
from ..fields.base import DataFieldBase, FieldBase
from ..tools.numba import jit
from ..tools.typing import ArrayLike
from ..trackers.base import TrackerCollectionDataType

if TYPE_CHECKING:
    from ..solvers.base import SolverBase
    from ..solvers.controller import TRangeType


TState = TypeVar("TState", FieldCollection, DataFieldBase)


class PDEBase(metaclass=ABCMeta):
    """base class for defining partial differential equations (PDEs)

    Custom PDEs can be implemented by subclassing :class:`PDEBase` to specify the
    evolution rate. In the simple case of deterministic PDEs, the methods
    :meth:`PDEBase.evolution_rate` and :meth:`PDEBase._make_pde_rhs_numba` need to be
    overwritten for supporting the `numpy` and `numba` backend, respectively.
    """

    diagnostics: dict[str, Any]
    """dict: Diagnostic information (available after the PDE has been solved)"""

    check_implementation: bool = True
    """bool: Flag determining whether numba-compiled functions should be checked against
    their numpy counter-parts. This can help with implementing a correct compiled
    version for a PDE class."""

    cache_rhs: bool = False
    """bool: Flag indicating whether the right hand side of the equation should be
    cached. If True, the same implementation is used in subsequent calls to `solve`.
    Note that this might lead to wrong results if the parameters of the PDE are changed
    after the first call. This option is thus disabled by default and should be used
    with care."""

    explicit_time_dependence: bool | None = None
    """bool: Flag indicating whether the right hand side of the PDE has an explicit
    time dependence."""

    complex_valued: bool = False
    """bool: Flag indicating whether the right hand side is a complex-valued PDE, which
    requires all involved variables to have complex data type."""

    def __init__(self, *, noise: ArrayLike = 0, rng: np.random.Generator | None = None):
        """
        Args:
            noise (float or :class:`~numpy.ndarray`):
                Variance of the additive Gaussian white noise that is supported for all
                PDEs by default. If set to zero, a deterministic partial differential
                equation will be solved. Different noise magnitudes can be supplied for
                each field in coupled PDEs.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used for stochastic simulations. Note that this random number generator
                is only used for numpy function, while compiled numba code uses the
                random number generator of numba. Moreover, in simulations using
                multiprocessing, setting the same generator in all processes might yield
                unintended correlations in the simulation results.

        Note:
            If more complicated noise structures are required, the methods
            :meth:`PDEBase.noise_realization` and
            :meth:`PDEBase._make_noise_realization_numba` need to be overwritten for the
            `numpy` and `numba` backend, respectively.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._cache: dict[str, Any] = {}
        self.diagnostics = {}
        self.noise = np.asanyarray(noise)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def is_sde(self) -> bool:
        """bool: flag indicating whether this is a stochastic differential equation

        The :class:`BasePDF` class supports additive Gaussian white noise, whose
        magnitude is controlled by the `noise` property. In this case, `is_sde` is
        `True` if `self.noise != 0`.
        """
        # check for self.noise, in case __init__ is not called in a subclass
        return hasattr(self, "noise") and np.any(self.noise != 0)  # type: ignore

    def make_modify_after_step(self, state: FieldBase) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to modify a state

        This function is applied to the state after each integration step when an
        explicit stepper is used. The default behavior is to not change the state.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be applied to a state to modify it and which returns a
            measure for the corrections applied to the state
        """

        def modify_after_step(state_data: np.ndarray) -> float:
            """no-op function"""
            return 0

        return modify_after_step

    @abstractmethod
    def evolution_rate(self, state: TState, t: float = 0) -> TState:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                The field at the current time point
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.base.FieldBase`:
                Field describing the evolution rate of the PDE
        """

    def _make_pde_rhs_numba(
        self, state: FieldBase, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function for evaluating the right hand side"""
        raise NotImplementedError("No backend `numba`")

    def check_rhs_consistency(
        self,
        state: TState,
        t: float = 0,
        *,
        tol: float = 1e-7,
        rhs_numba: Callable | None = None,
        **kwargs,
    ) -> None:
        """check the numba compiled right hand side versus the numpy variant

        Args:
            state (:class:`~pde.fields.FieldBase`):
                The state for which the evolution rates should be compared
            t (float):
                The associated time point
            tol (float):
                Acceptance tolerance. The check passes if the evolution rates differ by
                less then this value
            rhs_numba (callable):
                The implementation of the numba variant that is to be checked. If
                omitted, an implementation is obtained by calling
                :meth:`PDEBase._make_pde_rhs_numba_cached`.
        """
        # obtain evolution rate from the numpy implementation
        res_numpy = self.evolution_rate(state.copy(), t, **kwargs).data
        if not np.all(np.isfinite(res_numpy)):
            self._logger.warning(
                "The numpy implementation of the PDE returned non-finite values."
            )

        # obtain evolution rate from the numba implementation
        if rhs_numba is None:
            rhs_numba = self._make_pde_rhs_numba_cached(state, **kwargs)
        test_state = state.copy()
        res_numba = rhs_numba(test_state.data, t)
        if not np.all(np.isfinite(res_numba)):
            self._logger.warning(
                "The numba implementation of the PDE returned non-finite values."
            )

        # compare the two implementations
        msg = (
            "The numba compiled implementation of the right hand side is not "
            "compatible with the numpy implementation. This check can be disabled "
            "by setting the class attribute `check_implementation` to `False`."
        )
        np.testing.assert_allclose(
            res_numba, res_numpy, err_msg=msg, rtol=tol, atol=tol, equal_nan=True
        )

    def _make_pde_rhs_numba_cached(
        self, state: TState, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function for evaluating the right hand side

        This method implements caching and checking of the actual method, which is
        defined by overwriting the method `_make_pde_rhs_numba`.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        check_implementation = self.check_implementation

        if self.cache_rhs:
            # support caching of the right hand side
            grid_state = state.grid.state_serialized
            if self._cache.get("pde_rhs_numba_state") == grid_state:
                # cache was successful
                self._logger.info("Use compiled rhs from cache")
                check_implementation = False  # skip checking to save time
            else:
                # cache was not hit
                self._logger.info("Write compiled rhs to cache")
                self._cache["pde_rhs_numba_state"] = grid_state
                self._cache["pde_rhs_numba"] = self._make_pde_rhs_numba(state, **kwargs)
            rhs = self._cache["pde_rhs_numba"]

        else:
            # caching was skipped
            rhs = self._make_pde_rhs_numba(state, **kwargs)

        if rhs is None:
            raise RuntimeError("`_make_pde_rhs_numba` returned None")

        if check_implementation:
            self.check_rhs_consistency(state, rhs_numba=rhs, **kwargs)

        return rhs  # type: ignore

    def make_pde_rhs(
        self,
        state: TState,
        backend: Literal["auto", "numpy", "numba"] = "auto",
        **kwargs,
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """return a function for evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted values are 'numpy' and
                'numba'. Alternatively, 'auto' lets the code pick the optimal backend.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        if backend == "auto":
            try:
                rhs = self._make_pde_rhs_numba_cached(state, **kwargs)
            except NotImplementedError:
                backend = "numpy"
            else:
                rhs._backend = "numba"  # type: ignore

        if backend == "numpy":
            state = state.copy()

            def evolution_rate_numpy(state_data: np.ndarray, t: float) -> np.ndarray:
                """evaluate the rhs given only a state without the grid"""
                state.data = state_data
                return self.evolution_rate(state, t, **kwargs).data

            rhs = evolution_rate_numpy
            rhs._backend = "numpy"  # type: ignore

        elif backend == "numba":
            rhs = self._make_pde_rhs_numba_cached(state, **kwargs)
            rhs._backend = "numba"  # type: ignore

        elif backend != "auto":
            raise ValueError(
                f"Unknown backend `{backend}`. Possible values are ['auto', 'numpy', "
                "'numba']"
            )

        return rhs

    def noise_realization(
        self, state: TState, t: float = 0, *, label: str = "Noise realization"
    ) -> TState:
        """returns a realization for the noise

        Args:
            state (:class:`~pde.fields.ScalarField`):
                The scalar field describing the concentration distribution
            t (float):
                The current time point
            label (str):
                The label for the returned field

        Returns:
            :class:`~pde.fields.ScalarField`:
            Scalar field describing the evolution rate of the PDE
        """
        result: TState
        if self.is_sde:
            if isinstance(state, FieldCollection):
                # multiple fields with potentially different noise strengths
                result = state.copy(label=label)
                noises_var = np.broadcast_to(self.noise, len(state))

                for field, noise_var in zip(result, noises_var):
                    if noise_var == 0:
                        field.data.fill(0)
                    else:
                        field.data = field.random_normal(  # type: ignore
                            state.grid,
                            std=np.sqrt(noise_var),
                            scaling="physical",
                            dtype=state.dtype,
                            rng=self.rng,
                        )

            elif isinstance(state, DataFieldBase):
                # a single field
                noise_var = self.noise
                result = state.random_normal(
                    state.grid,
                    std=np.sqrt(self.noise),
                    scaling="physical",
                    label=label,
                    dtype=state.dtype,
                    rng=self.rng,
                )

            else:
                raise TypeError

        else:
            # no noise
            result = state.copy(label=label)
            result.data[:] = 0

        return result

    def _make_noise_realization_numba(
        self, state: TState, **kwargs
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """return a function for evaluating the noise term of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other
                information can be extracted

        Returns:
            Function determining the right hand side of the PDE
        """
        if self.is_sde:
            data_shape: tuple[int, ...] = state.data.shape
            noise_var: float
            cell_volume: Callable[[int], float] = state.grid.make_cell_volume_compiled(
                flat_index=True
            )
            assert state.dtype == float

            if isinstance(state, FieldCollection):
                # different noise strengths, assuming one for each field
                noises_var: np.ndarray = np.empty(data_shape[0])
                for n, noise_var in enumerate(np.broadcast_to(self.noise, len(state))):
                    noises_var[state._slices[n]] = noise_var

                @jit
                def noise_realization(state_data: np.ndarray, t: float) -> np.ndarray:
                    """helper function returning a noise realization"""
                    out = np.empty(data_shape)
                    for n in range(len(state_data)):
                        if noises_var[n] == 0:
                            out[n].fill(0)
                        else:
                            for i in range(state_data[n].size):
                                scale = noises_var[n] / cell_volume(i)
                                out[n].flat[i] = np.sqrt(scale) * np.random.randn()
                    return out

            else:
                # a single noise value is given for all fields
                noise_var = float(self.noise)

                @jit
                def noise_realization(state_data: np.ndarray, t: float) -> np.ndarray:
                    """helper function returning a noise realization"""
                    out = np.empty(state_data.shape)
                    for i in range(state_data.size):
                        scale = noise_var / cell_volume(i)
                        out.flat[i] = np.sqrt(scale) * np.random.randn()
                    return out

        else:

            @jit
            def noise_realization(state_data: np.ndarray, t: float) -> None:
                """helper function returning a noise realization"""
                return None

        return noise_realization  # type: ignore

    def _make_sde_rhs_numba(
        self, state: TState, **kwargs
    ) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
        """return a function for evaluating the noise term of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function determining the right hand side of the PDE
        """
        evolution_rate = self._make_pde_rhs_numba_cached(state, **kwargs)
        noise_realization = self._make_noise_realization_numba(state, **kwargs)

        @jit
        def sde_rhs(state_data: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
            """compiled helper function returning a noise realization"""
            return (evolution_rate(state_data, t), noise_realization(state_data, t))

        return sde_rhs  # type: ignore

    def _make_sde_rhs_numba_cached(
        self, state: TState, **kwargs
    ) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
        """create a compiled function for evaluating the noise term of the PDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        This method implements caching and checking of the actual method, which is
        defined by overwriting the method `_make_pde_rhs_numba`.
        """
        if self.cache_rhs:
            # support caching of the noise term
            grid_state = state.grid.state_serialized
            if self._cache.get("sde_rhs_numba_state") == grid_state:
                # cache was successful
                self._logger.info("Use compiled noise term from cache")
            else:
                # cache was not hit
                self._logger.info("Write compiled noise term to cache")
                self._cache["sde_rhs_numba_state"] = grid_state
                self._cache["sde_rhs_numba"] = self._make_sde_rhs_numba(state, **kwargs)
            sde_rhs = self._cache["sde_rhs_numba"]

        else:
            # caching was skipped
            sde_rhs = self._make_sde_rhs_numba(state, **kwargs)

        return sde_rhs  # type: ignore

    def make_sde_rhs(
        self,
        state: TState,
        backend: Literal["auto", "numpy", "numba"] = "auto",
        **kwargs,
    ) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
        """return a function for evaluating the right hand side of the SDE

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted
            backend (str):
                Determines how the function is created. Accepted values are 'numpy' and
                'numba'. Alternatively, 'auto' lets the code pick the optimal backend.

        Returns:
            Function determining the deterministic part of the right hand side of the
            PDE together with a noise realization.
        """
        if backend == "auto":
            try:
                sde_rhs = self._make_sde_rhs_numba_cached(state, **kwargs)
            except NotImplementedError:
                backend = "numpy"
            else:
                sde_rhs._backend = "numba"  # type: ignore
                return sde_rhs

        if backend == "numba":
            sde_rhs = self._make_sde_rhs_numba_cached(state, **kwargs)
            sde_rhs._backend = "numba"  # type: ignore

        elif backend == "numpy":
            state = state.copy()

            def sde_rhs(
                state_data: np.ndarray, t: float
            ) -> tuple[np.ndarray, np.ndarray]:
                """evaluate the rhs given only a state without the grid"""
                state.data = state_data
                return (
                    self.evolution_rate(state, t, **kwargs).data,
                    self.noise_realization(state, t, **kwargs).data,
                )

            sde_rhs._backend = "numpy"  # type: ignore

        else:
            raise ValueError(f"Unsupported backend `{backend}`")

        return sde_rhs

    def solve(
        self,
        state: TState,
        t_range: TRangeType,
        dt: float | None = None,
        tracker: TrackerCollectionDataType = "auto",
        *,
        solver: str | SolverBase = "explicit",
        ret_info: bool = False,
        **kwargs,
    ) -> None | TState | tuple[TState | None, dict[str, Any]]:
        """solves the partial differential equation

        The method constructs a suitable solver (:class:`~pde.solvers.base.SolverBase`)
        and controller (:class:`~pde.controller.Controller`) to advance the state over
        the temporal range specified by `t_range`. This method only exposes the most
        common functions, so explicit construction of these classes might offer more
        flexibility.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                The initial state (which also defines the spatial grid).
            t_range (float or tuple):
                Sets the time range for which the PDE is solved. This should typically
                be a tuple of two numbers, `(t_start, t_end)`, specifying the initial
                and final time of the simulation. If only a single value is given, it is
                interpreted as `t_end` and the time range is `(0, t_end)`.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen. If an adaptive stepper is used
                (supported by :class:`~pde.solvers.ScipySolver` and
                :class:`~pde.solvers.ExplicitSolver`), `dt` sets the initial time step.
            tracker:
                Defines trackers that process the state of the simulation at specified
                times. A tracker is either an instance of
                :class:`~pde.trackers.base.TrackerBase` or a string identifying a
                tracker (possible identifiers can be obtained by calling
                :func:`~pde.trackers.base.get_named_trackers`). Multiple trackers can be
                specified as a list. The default value `auto` checks the state for
                consistency (tracker 'consistency') and displays a progress bar (tracker
                'progress') when :mod:`tqdm` is installed. More general trackers are
                defined in :mod:`~pde.trackers`, where all options are explained in
                detail. In particular, the time points where the tracker analyzes data
                can be chosen when creating a tracker object explicitly.
            solver (:class:`~pde.solvers.base.SolverBase` or str):
                Specifies the method for solving the differential equation. This can
                either be an instance of :class:`~pde.solvers.base.SolverBase` or a
                descriptive name like 'explicit' or 'scipy'. The valid names are given
                by :meth:`pde.solvers.registered_solvers`. Details of the solvers and
                additional features (like adaptive time steps) are explained in
                :mod:`~pde.solvers`.
            ret_info (bool):
                Flag determining whether diagnostic information about the solver process
                should be returned. Note that the same information is also available
                as the :attr:`~PDEBase.diagnostics` attribute.
            **kwargs:
                Additional keyword arguments are forwarded to the solver class chosen
                with the `solver` argument. In particular,
                :class:`~pde.solvers.explicit.ExplicitSolver` supports several `schemes`
                and an adaptive stepper can be enabled using :code:`adaptive=True`.
                Conversely, :class:`~pde.solvers.ScipySolver` accepts the additional
                arguments of :func:`scipy.integrate.solve_ivp`.

        Returns:
            :class:`~pde.fields.base.FieldBase`:
            The state at the final time point. If `ret_info == True`, a tuple with the
            final state and a dictionary with additional information is returned. Note
            that `None` instead of a field is returned in multiprocessing simulations if
            the current node is not the main MPI node.
        """
        from ..solvers import Controller
        from ..solvers.base import SolverBase  # @Reimport

        # create solver instance
        if callable(solver):
            solver_obj = solver(pde=self, **kwargs)
            if not isinstance(solver_obj, SolverBase):
                self._logger.warning("Solver is not an instance of `SolverBase`.")

        elif isinstance(solver, str):
            solver_obj = SolverBase.from_name(solver, pde=self, **kwargs)

        else:
            raise TypeError(f"Solver {solver} is not supported")

        # create controller instance
        controller = Controller(solver_obj, t_range=t_range, tracker=tracker)

        # run the simulation
        final_state = controller.run(state, dt)

        # copy diagnostic information to the PDE instance
        if hasattr(self, "diagnostics"):
            self.diagnostics.update(controller.diagnostics)
        else:
            self.diagnostics = copy.copy(controller.diagnostics)

        if ret_info:
            # return a copy of the diagnostic information so it will not be overwritten
            # by a repeated call to `solve()`.
            return final_state, copy.deepcopy(self.diagnostics)
        else:
            return final_state


def expr_prod(factor: float, expression: str) -> str:
    """helper function for building an expression with an (optional) pre-factor

    Args:
        factor (float): The value of the prefactor
        expression (str): The remaining expression

    Returns:
        str: The expression with the factor appended if necessary
    """
    if factor == 0:
        return "0"
    elif factor == 1:
        return expression
    elif factor == -1:
        return "-" + expression
    else:
        return f"{factor:g} * {expression}"
