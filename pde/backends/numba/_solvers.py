"""Implements numba-accelerated solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from ...solvers import AdamsBashforthSolver, EulerSolver
from ...solvers.base import (
    AdaptiveSolverBase,
    AdaptiveStepperType,
    FixedStepperType,
    SolverBase,
    _make_dt_adjuster,
)
from ...tools.typing import NumericArray, StepperHook, TField
from .utils import jit

if TYPE_CHECKING:
    from ...tools.math import OnlineStatistics

SingleStepType = Callable[[NumericArray, float], None]


def _make_post_step_hook(solver: SolverBase, state: TField) -> StepperHook:
    """Create a callable that executes the PDE's post-step hook.

    The returned function calls the post-step hook provided by the PDE (if any)
    after each completed time step. If the PDE implements make_post_step_hook,
    this method attempts to obtain both the hook function and an initial value
    for the hook's mutable data by calling
    `post_step_hook, post_step_data_init = self.pde.make_post_step_hook(state, backend)`
    The initial data is stored on the solver instance as `self._post_step_data_init`
    (copied to ensure mutability) and will be passed to the hook when the stepper
    is executed.

    If no hook is provided by the PDE (i.e., `make_post_step_hook` raises
    :class:`NotImplementedError`) or if the solver's `_use_post_step_hook` flag
    is `False`, a no-op hook is returned and `self._post_step_data_init` is set
    to `None`.

    The hook returned by this method always conforms to the signature
    `(state_data: numpy.ndarray, t: float, post_step_data: numpy.ndarray) -> None`
    and is suitable for JIT compilation where supported.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the hook is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            Example field providing the array shape and grid information required
            by the PDE when constructing the post-step hook.

    Returns:
        callable:
            A function that invokes the PDE's post-step hook (or a no-op) with the
            signature described above.
    """
    # get uncompiled post_step_hook
    post_step_hook = solver._make_post_step_hook(state)

    # compile post_step_hook
    post_step_data_type = nb.typeof(solver._post_step_data_init)
    signature_hook = (nb.typeof(state.data), nb.float64, post_step_data_type)
    post_step_hook = jit(signature_hook)(post_step_hook)

    solver._logger.debug("Compiled post-step hook")
    return post_step_hook  # type: ignore


def _make_fixed_stepper(
    solver: SolverBase, state: TField, dt: float
) -> FixedStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted
        dt (float):
            Time step of the explicit stepping.
    """
    # get compiled version of a single step
    single_step = solver._make_single_step_fixed_dt(state, dt)
    single_step_signature = (nb.typeof(state.data), nb.double)
    single_step = jit(single_step_signature)(single_step)
    post_step_hook = _make_post_step_hook(solver, state)

    # provide compiled function doing all steps
    fixed_stepper_signature = (
        nb.typeof(state.data),
        nb.double,
        nb.int_,
        nb.typeof(solver._post_step_data_init),
    )

    @jit(fixed_stepper_signature)
    def fixed_stepper(
        state_data: NumericArray, t_start: float, steps: int, post_step_data
    ) -> float:
        """Perform `steps` steps with fixed time steps."""
        for i in range(steps):
            # calculate the right hand side
            t = t_start + i * dt
            single_step(state_data, t)
            post_step_hook(state_data, t, post_step_data)

        return t + dt

    return fixed_stepper  # type: ignore


def _make_adams_bashforth_stepper(
    solver: AdamsBashforthSolver, state: TField, dt: float
) -> FixedStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.adams_bashforth.AdamsBashforthSolver`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted
        dt (float):
            Time step of the explicit stepping.
    """
    if solver.pde.is_sde:
        msg = "Deterministic Adams-Bashforth does not support stochastic equations"
        raise RuntimeError(msg)

    rhs_pde = solver.pde.make_pde_rhs(state, backend=solver.backend)
    post_step_hook = _make_post_step_hook(solver, state)
    sig_single_step = (nb.typeof(state.data), nb.double, nb.typeof(state.data))

    @jit(sig_single_step)
    def single_step(
        state_data: NumericArray, t: float, state_prev: NumericArray
    ) -> None:
        """Perform a single Adams-Bashforth step."""
        rhs_prev = rhs_pde(state_prev, t - dt).copy()
        rhs_cur = rhs_pde(state_data, t)
        state_prev[:] = state_data  # save the previous state
        state_data += dt * (1.5 * rhs_cur - 0.5 * rhs_prev)

    # allocate memory to store the state of the previous time step
    state_prev = np.empty_like(state.data)
    init_state_prev = True

    def fixed_stepper(
        state_data: NumericArray, t_start: float, steps: int, post_step_data
    ) -> float:
        """Perform `steps` steps with fixed time steps."""
        nonlocal state_prev, init_state_prev

        if init_state_prev:
            # initialize the state_prev with an estimate of the previous step
            state_prev[:] = state_data - dt * rhs_pde(state_data, t_start)
            init_state_prev = False

        for i in range(steps):
            # calculate the right hand side
            t = t_start + i * dt
            single_step(state_data, t, state_prev)
            post_step_hook(state_data, t, post_step_data=post_step_data)

        return t + dt

    solver._logger.info("Init explicit Adams-Bashforth stepper with dt=%g", dt)

    return fixed_stepper


def make_fixed_stepper(
    solver: SolverBase, state: TField, dt: float
) -> FixedStepperType:
    """Return a stepper function using an explicit scheme with fixed time steps.

    Args:
        solver (:class:`~pde.solvers.base.SolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted
        dt (float):
            Time step of the explicit stepping.
    """
    if isinstance(solver, AdamsBashforthSolver):
        return _make_adams_bashforth_stepper(solver, state, dt)
    return _make_fixed_stepper(solver, state, dt)


def _make_adaptive_stepper_general(
    solver: AdaptiveSolverBase, state: TField
) -> AdaptiveStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    # obtain functions determining how the PDE is evolved
    single_step_error = solver._make_single_step_error_estimate(state)
    signature_single_step = (nb.typeof(state.data), nb.double, nb.double)
    single_step_error = jit(signature_single_step)(single_step_error)
    post_step_hook = _make_post_step_hook(solver, state)
    sync_errors = solver._backend_obj.make_mpi_synchronizer(operator="MAX")

    # obtain auxiliary functions
    signature = (nb.double, nb.double)
    adjust_dt = jit(signature)(_make_dt_adjuster(solver.dt_min, solver.dt_max))
    tolerance = solver.tolerance
    dt_min = solver.dt_min

    def adaptive_stepper(
        state_data: NumericArray,
        t_start: float,
        t_end: float,
        dt_init: float,
        dt_stats: OnlineStatistics | None = None,
        post_step_data=None,
    ) -> tuple[float, float, int]:
        """Adaptive stepper that advances the state in time."""
        dt_opt = dt_init
        t = t_start
        steps = 0
        while True:
            # use a smaller (but not too small) time step if close to t_end
            dt_step = max(min(dt_opt, t_end - t), dt_min)

            # try two different step sizes to estimate errors
            new_state, error = single_step_error(state_data, t, dt_step)

            error_rel = error / tolerance  # normalize error to given tolerance
            # synchronize the error between all processes (necessary for MPI)
            error_rel = sync_errors(error_rel)

            # do the step if the error is sufficiently small
            if error_rel <= 1:
                steps += 1
                t += dt_step
                state_data[...] = new_state
                post_step_hook(state_data, t, post_step_data)

                if dt_stats is not None:
                    dt_stats.add(dt_step)

            if t < t_end:
                # adjust the time step and continue (happens in every MPI process)
                dt_opt = adjust_dt(dt_step, error_rel)
            else:
                break  # return to the controller

        return t, dt_opt, steps

    if not nb.config.DISABLE_JIT:
        # do the compilation only when JIT is actually being done. This might be
        # disabled for debugging numba code or for determining test coverage
        signature_stepper = (
            nb.typeof(state.data),
            nb.double,
            nb.double,
            nb.double,
            nb.typeof(solver.info["dt_statistics"]),
            nb.typeof(solver._post_step_data_init),
        )
        adaptive_stepper = jit(signature_stepper)(adaptive_stepper)

    solver._logger.info("Initialized adaptive stepper")
    return adaptive_stepper


def _make_adaptive_stepper_euler(
    solver: EulerSolver, state: TField
) -> AdaptiveStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.explicit.EulerSolver`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    if nb.config.DISABLE_JIT:
        # this can be useful to debug numba implementations and for test coverage checks
        return solver._make_adaptive_stepper(state)
    # create compiled function for adjusting the time step
    adjust_dt = _make_dt_adjuster(solver.dt_min, solver.dt_max)
    adjust_signature = (nb.double, nb.double)
    adjust_dt = jit(adjust_signature)(adjust_dt)

    # create the adaptive stepper and compile it
    stepper = solver._make_adaptive_stepper(
        state,
        post_step_hook=_make_post_step_hook(solver, state),
        adjust_dt=adjust_dt,
    )
    signature = (
        nb.typeof(state.data),
        nb.double,
        nb.double,
        nb.double,
        nb.typeof(solver.info["dt_statistics"]),
        nb.typeof(solver._post_step_data_init),
    )
    return jit(signature)(stepper)  # type: ignore


def make_adaptive_stepper(
    solver: AdaptiveSolverBase, state: TField
) -> AdaptiveStepperType:
    """Return a stepper function using an explicit scheme.

    Args:
        solver (:class:`~pde.solvers.base.AdaptiveSolverBase`):
            The solver instance, which determines how the stepper is constructed
        state (:class:`~pde.fields.base.FieldBase`):
            An example for the state from which the grid and other information can
            be extracted

    Returns:
        Function that can be called to advance the `state` from time `t_start` to
        time `t_end`. The function call signature is `(state: numpy.ndarray,
        t_start: float, t_end: float)`
    """
    if isinstance(solver, EulerSolver):
        return _make_adaptive_stepper_euler(solver, state)
    return _make_adaptive_stepper_general(solver, state)
