"""Base class for defining partial differential equations.

.. autosummary::
   :nosignatures:

   PDEBase
   SDEBase

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..backends import BackendBase, get_backend
from ..fields import FieldCollection
from ..fields.datafield_base import DataFieldBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..fields.base import FieldBase
    from ..solvers.base import SolverBase
    from ..solvers.controller import TRangeType
    from ..tools.typing import (
        ArrayLike,
        NumericArray,
        StepperHook,
        TField,
        TNativeArray,
    )
    from ..trackers.base import TrackerCollectionDataType

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for PDEs."""


class PDEBase(metaclass=ABCMeta):
    """Base class for defining deterministic partial differential equations (PDEs)

    Custom PDEs can be implemented by subclassing :class:`PDEBase` to specify the
    evolution rate. In the simplest case, only the :meth:`PDEBase.evolution_rate` needs
    to be implemented to support the `numpy` backend. Other backends require overwriting
    the :meth:`PDEBase.make_evolution_rate`.
    """

    diagnostics: dict[str, Any]
    """dict: Diagnostic information (available after the PDE has been solved)"""

    check_implementation: bool = True
    """bool: Flag determining whether numba-compiled functions should be checked against
    their numpy counter-parts. This can help with implementing a correct compiled
    version for a PDE class.

    Warning: This flag is deprecated since 2025-12-13 and this check will not be
    performed automatically anymore.
    """

    cache_rhs: bool = False
    """bool: Flag indicating whether the right hand side of the equation should be
    cached. If True, the same implementation is used in subsequent calls to `solve`.
    Note that the cache is only invalidated when the grid of the underlying state
    changes. Consequently, the simulation might lead to wrong results if the parameters
    of the PDE are changed after the first call. This option is thus disabled by default
    and should be used with care.

    Warning: This flag is deprecated since 2025-12-13 and caching is not implemented
    anymore.
    """

    explicit_time_dependence: bool | None = None
    """bool: Flag indicating whether the right hand side of the PDE has an explicit
    time dependence."""

    complex_valued: bool = False
    """bool: Flag indicating whether the right hand side is a complex-valued PDE, which
    requires all involved variables to have complex data type."""

    _mpi_synchronization: bool = False
    """bool: Flag indicating whether the PDE will be solved on multiple nodes using MPI.
    This flag will be set by the solver. If it is true and the PDE requires global
    values in its evaluation, the synchronization between nodes needs to be handled. In
    many cases, PDEs are defined locally and no such synchronization is necessary. Note
    that the virtual points at the boundaries are synchronized automatically."""

    _logger: logging.Logger

    def __init__(self, *, rng: np.random.Generator | None = None):
        """
        Args:
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used for stochastic simulations. Note that this random number generator
                is only used for numpy functions, while other backends might not use it.
                Moreover, in simulations using multiprocessing, setting the same
                generator in all processes might yield unintended correlations in the
                simulation results.
        """
        self._cache: dict[str, Any] = {}
        self.diagnostics = {}
        self.rng = np.random.default_rng(rng)

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses.

        Args:
            **kwargs:
                Additional keyword arguments passed to the superclass
        """
        super().__init_subclass__(**kwargs)
        # create logger for this specific PDE class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_cache"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}

    @property
    def is_sde(self) -> bool:
        """bool: flag indicating whether this is a stochastic differential equation"""
        return False

    def make_post_step_hook(
        self, state: FieldBase, backend: str | BackendBase = "numpy"
    ) -> tuple[StepperHook, Any]:
        """Returns a function that is called after each step.

        This function receives three arguments: the current state as a numpy array, the
        current time point, and a numpy array that can store data for the hook function.
        The function must return the state data and the hook data, which it can both
        modify in place.

        The hook can also be used to abort the simulation when a user-defined condition
        is met by raising `StopIteration`. Note that this interrupts the inner-most
        loop, so that some final information might be still reflect the values they
        assumed at the last tracker interrupt. Additional information (beside the
        current state) should be returned by the `post_step_data`. Note that raising
        `StopIteration` only works for some backends.

        Example:
            The following code provides an example that creates a hook function that
            limits the state to a maximal value of 1 and keeps track of the total
            correction that is applied. This is achieved using `post_step_data`, which
            is initialized with the second value (0) returned by the method and
            incremented each time the hook is called.

            .. code-block:: python

                def make_post_step_hook(self, state, backend):
                    def post_step_hook(state_data, t, post_step_data):
                        i = state_data > 1  # get violating entries
                        overshoot = (state_data[i] - 1).sum()  # get total correction
                        state_data[i] = 1  # limit data entries
                        post_step_data += overshoot  # accumulate total correction
                        return state_data, post_step_data

                    return post_step_hook, 0.0  # hook function and initial value

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            backend (str):
                Determines how the function is created (like 'numpy' and 'numba')

        Returns:
            tuple: The first entry is the function that implements the hook. The second
                entry gives the initial data that is used as auxiliary data in the hook.
                This can be `None` if no data is used.
        """
        raise NotImplementedError

    @abstractmethod
    def evolution_rate(self, state: TField, t: float = 0) -> TField:
        """Evaluate the right hand side of the PDE.

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
        self, state: FieldBase
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Create a compiled function for evaluating the right hand side."""
        # deprecated on 2025-12-13
        warnings.warn(
            "Method `_make_pde_rhs_numba` is deprecated in favor of "
            "`make_evolution_rate`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.make_evolution_rate(state, backend=get_backend("numba"))

    def check_rhs_consistency(
        self,
        rhs_implementation: Callable,
        state: TField,
        t: float = 0,
        *,
        tol: float = 1e-7,
    ) -> None:
        """Checks an implementation of the right hand side versus the numpy variant.

        Args:
            rhs_implementation (callable):
                The implementation of the numba variant that is to be checked.
            state (:class:`~pde.fields.FieldBase`):
                The state for which the evolution rates should be compared
            t (float):
                The associated time point
            tol (float):
                Acceptance tolerance. The check passes if the evolution rates differ by
                less then this value
        """
        # obtain evolution rate from the numpy implementation
        res_numpy = self.evolution_rate(state.copy(), t).data
        if not np.all(np.isfinite(res_numpy)):
            self._logger.warning(
                "The numpy implementation of the PDE returned non-finite values."
            )

        # obtain evolution rate from the numba implementation
        test_state = state.copy()
        res_numba = rhs_implementation(test_state.data, t)
        if not np.all(np.isfinite(res_numba)):
            self._logger.warning(
                "The tested implementation of the PDE returned non-finite values."
            )

        # compare the two implementations
        msg = (
            "The tested compiled implementation of the right hand side is not "
            "compatible with the numpy implementation. Additional information is "
            "available in `diagnostics['check']`. This check can be disabled by "
            "setting the class attribute `check_implementation` to `False`."
        )
        try:
            np.testing.assert_allclose(
                res_numba, res_numpy, err_msg=msg, rtol=tol, atol=tol, equal_nan=True
            )
        except AssertionError:
            # convert the two right hand sides into respective fields
            field_rhs_numpy = state.copy(label="RHS, numpy")
            field_rhs_numpy.data = res_numpy
            field_rhs_numba = state.copy(label="RHS, numba")
            field_rhs_numba.data = res_numba
            # store diagnostic information for debugging
            self.diagnostics["check"] = {
                "state": state,
                "rhs_numpy": field_rhs_numpy,
                "rhs_numba": field_rhs_numba,
            }
            # re-raise the exception
            raise

    def _make_pde_rhs_numba_cached(
        self, state: TField
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Create a compiled function for evaluating the right hand side.

        This method implements caching and checking of the actual method, which is
        defined by overwriting the method `make_pde_rhs_numba`.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        # deprecated on 2025-12-13
        # If this deprecation is removed, we can also get rid of the properties
        # `cache_rhs` and `check_implementation`
        warnings.warn(
            "Method `_make_pde_rhs_numba_cached` is deprecated. Use the uncached "
            "method `make_evolution_rate` instead",
            DeprecationWarning,
            stacklevel=2,
        )

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
                self._cache["pde_rhs_numba"] = self._make_pde_rhs_numba(state)
            rhs = self._cache["pde_rhs_numba"]

        else:
            # caching was skipped
            rhs = self._make_pde_rhs_numba(state)

        if rhs is None:
            msg = "`make_pde_rhs_numba` returned None"
            raise RuntimeError(msg)

        if check_implementation:
            self.check_rhs_consistency(rhs_implementation=rhs, state=state)

        return rhs  # type: ignore

    def determine_backend(
        self, state: TField, backend: str | BackendBase = "auto"
    ) -> BackendBase:
        """Returns backend that will be chosen automatically for this PDE.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Information about which backend to choose. The special value `auto`
                tries various backends and returns one for which the evolution rate is
                implemented for this PDE.

        Returns:
            str: The backend used automatically
        """
        if isinstance(backend, BackendBase):
            return backend  # backend has already been selected
        if backend != "auto":
            return get_backend(backend)  # load the respective backend

        # choose backend automatically by trial and error to see which one works
        for backend in ["numba", "torch", "numpy"]:
            # TODO: Could first add a check whether module is available; Issue #762
            try:
                self.make_pde_rhs(state, backend=backend)
            except (NotImplementedError, ModuleNotFoundError) as err:
                self._logger.info("Using backend `%s` failed: %s", backend, str(err))
            else:
                break  # found a suitable backend
        else:
            msg = "Could not select a suitable backend"
            raise RuntimeError(msg)
        return get_backend(backend)

    def make_pde_rhs(
        self,
        state: TField,
        backend: str | BackendBase = "auto",
    ) -> Callable[[TNativeArray, float], TNativeArray]:
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                The backend that is used to create the function. The special value
                `numpy` uses the method `evaluation_rate`. Other backends are only
                available if `make_evolution_rate` is defined for the PDE. If this is
                the case, the special value `auto` selects the `numba` backend,
                otherwise it defaults to `numpy`.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        # determine a suitable backend for the implementation
        backend = self.determine_backend(state, backend)

        # get a function evaluating the rhs of the PDE
        return backend.make_pde_rhs(self, state)

    def make_evolution_rate(
        self, state, backend: BackendBase
    ) -> Callable[[TNativeArray, float], TNativeArray]:
        """Return function evaluating right hand side of the PDE using given backend.

        Note:
            This factory function must return a function that processes fields stored
            in the native format of the backend. For instance, a function returned for
            the `jax` backend must deal with :class:`jax.Array` objects.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines the backend.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        raise NotImplementedError

    def solve(
        self,
        state: TField,
        t_range: TRangeType,
        dt: float | None = None,
        tracker: TrackerCollectionDataType = "auto",
        *,
        backend: str | BackendBase = "auto",
        solver: str | SolverBase = "euler",
        ret_info: bool = False,
        **kwargs,
    ) -> None | TField | tuple[TField | None, dict[str, Any]]:
        """Solves the partial differential equation.

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
                Time step of the chosen stepping scheme. If `None`, the solver chooses
                a default value when constructing the stepping function. If adaptive
                stepping is enabled (e.g., supported by
                :class:`~pde.solvers.EulerSolver`), `dt` sets the initial time step.
            tracker:
                Defines trackers that process the state of the simulation at specified
                times. A tracker is either an instance of
                :class:`~pde.trackers.base.TrackerBase` or a string identifying a
                tracker (possible identifiers can be obtained by calling
                :func:`~pde.trackers.registered_trackers`). Multiple trackers can be
                specified as a list. The default value `auto` checks the state for
                consistency (tracker 'consistency') and displays a progress bar (tracker
                'progress') when :mod:`tqdm` is installed. More general trackers are
                defined in :mod:`~pde.trackers`, where all options are explained in
                detail. In particular, the time points where the tracker analyzes data
                can be chosen when creating a tracker object explicitly.
            backend (str):
                Determines how the function is created. Accepted values are 'numpy' and
                'numba'. Alternatively, 'auto' lets the code pick the optimal backend.
            solver (:class:`~pde.solvers.base.SolverBase` or str):
                Specifies the persistent numerical strategy used for solving the
                differential equation. This can either be a solver factory or a
                descriptive name like 'explicit' or 'scipy'. The valid names are given
                by :meth:`pde.solvers.registered_solvers`. Details of the solver
                classes and additional features (like adaptive stepping) are explained
                in :mod:`~pde.solvers`.
            ret_info (bool):
                Flag determining whether diagnostic information about the solver and
                stepping process should be returned. Note that the same information is
                also available as the :attr:`~PDEBase.diagnostics` attribute.
            **kwargs:
                Additional keyword arguments are forwarded to the solver chosen with
                the `solver` argument. In particular, adaptive stepping can often be
                enabled using :code:`adaptive=True`.

        Returns:
            :class:`~pde.fields.base.FieldBase`:
            The state at the final time point. If `ret_info == True`, a tuple with the
            final state and a dictionary with additional information is returned. Note
            that `None` instead of a field is returned in multiprocessing simulations if
            the current node is not the main MPI node.
        """
        from ..solvers import Controller
        from ..solvers.base import SolverBase

        # create solver instance
        if callable(solver):
            solver_obj = solver(pde=self, backend=backend, **kwargs)
            if not isinstance(solver_obj, SolverBase):
                self._logger.warning("Solver is not an instance of `SolverBase`.")

        elif isinstance(solver, str):
            if solver in {"euler", "explicit", "explicit_mpi", "runge-kutta"}:
                # Use an adaptive solver in the default case of an explicit solver
                # when no time step is specified and use a fixed time step otherwise
                kwargs.setdefault("adaptive", dt is None)
            solver_obj = SolverBase.from_name(
                solver, pde=self, backend=backend, **kwargs
            )

        elif isinstance(solver, SolverBase):
            msg = "`solver` must be a class not an instance"
            raise TypeError(msg)

        else:
            msg = f"Solver {solver} is not supported"
            raise TypeError(msg)

        # create controller instance
        controller = Controller(solver_obj, t_range=t_range, tracker=tracker)

        # run the simulation
        try:
            final_state = controller.run(state, dt)
        finally:
            # copy diagnostic information to the PDE instance
            if hasattr(self, "diagnostics"):
                self.diagnostics.update(controller.diagnostics)
            else:
                self.diagnostics = copy.copy(controller.diagnostics)

        if ret_info:
            # return a copy of the diagnostic information so it will not be overwritten
            # by a repeated call to `solve()`.
            return final_state, copy.deepcopy(self.diagnostics)
        return final_state


class SDEBase(PDEBase):
    """Base class for defining stochastic partial differential equations (SDEs)

    Custom PDEs can be implemented by subclassing :class:`SDEBase` to specify the
    evolution rate and an associated noise realization. Overwrite
    :meth:`_make_noise_realization` (together with :meth:`PDEBase.make_evolution_rate`)
    to support all backends.
    """

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
                is only used for numpy functions, while other backends might not use it.
                Moreover, in simulations using multiprocessing, setting the same
                generator in all processes might yield unintended correlations in the
                simulation results.

        Note:
            If more complicated noise structures are required, overwrite
            :meth:`SDEBase.make_noise_variance` to provide a custom noise variance for
            all backends.
        """
        super().__init__(rng=rng)
        self.noise = np.asanyarray(noise)

    @property
    def is_sde(self) -> bool:
        """bool: flag indicating whether this is a stochastic differential equation

        The :class:`SDEBase` class supports additive Gaussian white noise, whose
        magnitude is controlled by the `noise` property. In this case, `is_sde` is
        `True` if `self.noise != 0`.
        """
        # check for self.noise, but do not assume it is defined in case __init__ is not
        # called in a subclass
        noise = getattr(self, "noise", 0)
        has_noise_var = not np.allclose(noise, 0, atol=1e-14)
        return has_noise_var or hasattr(self, "_make_noise_realization")

    @overload
    def make_noise_variance(
        self, state: TField, *, backend: BackendBase[TNativeArray]
    ) -> Callable[[TNativeArray, float], TNativeArray]: ...

    @overload
    def make_noise_variance(
        self,
        state: TField,
        *,
        backend: BackendBase[TNativeArray],
        ret_diff: Literal[False],
    ) -> Callable[[TNativeArray, float], TNativeArray]: ...

    @overload
    def make_noise_variance(
        self,
        state: TField,
        *,
        backend: BackendBase[TNativeArray],
        ret_diff: Literal[True],
    ) -> Callable[[TNativeArray, float], tuple[TNativeArray, TNativeArray]]: ...

    def make_noise_variance(
        self,
        state: TField,
        *,
        backend: BackendBase[TNativeArray],
        ret_diff: bool = False,
    ) -> Callable[
        [TNativeArray, float], TNativeArray | tuple[TNativeArray, TNativeArray]
    ]:
        """Make function that calculates noise variance.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines the backend.
            ret_diff (bool):
                Determines whether only the noise variance or also its derivative with
                respect to the field at this position is returned.

        Returns:
            A function with signature (state_data, t) that either returns just the noise
            variance or also its derivative, depending on `ret_diff`.
        """
        grid = state.grid
        if isinstance(state, DataFieldBase):
            # support different variance for each element in potential tensor
            noise_vars = np.broadcast_to(self.noise, state.data_shape)

        elif isinstance(state, FieldCollection):
            # support different variance for each field
            noise_var = np.broadcast_to(self.noise, len(state))
            noise_vars = np.empty(state.data.shape[0])
            for i, var in enumerate(noise_var):
                noise_vars[state._slices[i]] = var

        else:
            raise TypeError

        noise_vars = noise_vars.reshape(state.data_shape + (1,) * grid.num_axes)
        noise_vars_native: TNativeArray = backend.numpy_to_native(noise_vars)

        if ret_diff:
            noise_vars_diff_native = backend.numpy_to_native(np.zeros_like(noise_vars))

            def noise_variance_diff(
                state_data: TNativeArray, t: float
            ) -> tuple[TNativeArray, TNativeArray]:
                """Calculates noise variance and its derivative."""
                return noise_vars_native, noise_vars_diff_native

            return noise_variance_diff

        def noise_variance(state_data: TNativeArray, t: float) -> TNativeArray:
            """Calculates noise variance."""
            return noise_vars_native

        return noise_variance


def expr_prod(factor: float, expression: str) -> str:
    """Helper function for building an expression with an (optional) pre-factor.

    Args:
        factor (float):
            The value of the prefactor
        expression (str):
            The remaining expression

    Returns:
        str: The expression with the factor appended if necessary
    """
    if factor == 0:
        return "0"
    if factor == 1:
        return expression
    if factor == -1:
        return "-" + expression
    return f"{factor:g} * {expression}"
